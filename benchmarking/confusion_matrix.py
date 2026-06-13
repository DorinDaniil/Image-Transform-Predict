"""
Token-level confusion-matrix benchmark (simplified).

For N images (default 5000, quickdraw excluded) apply each atomic token
independently. target = applied token, prediction = the model's FIRST output
token, or "empty" if the model produced nothing. One matrix per custom model.

NOTE: each token is applied with the augmentations from the recall test
(make_recall_aug_apply_fn), which is the default. Pass apply_token_fn only if
you want to override it. Grayscale images are auto-skipped (color images only).

Assumes `SimpleDomainNetDataset` is importable (same project as your benchmarks).
"""

import io
import sys
import json
import random
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

src_path = '/mnt/DATA2/dorin/Image-Transform-Predict/src'
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from dataset import SimpleDomainNetDataset


ATOMIC_TOKENS = [
    "noop", "grayscale", "rotate_90", "rotate_180", "rotate_270",
    "color_jitter", "noise_adding", "crop",
    "horizontal_flip", "vertical_flip",
    "jpeg_artefacts",
]


def make_transformer_apply_fn(transformer):
    """Wrap ImageTransformer.apply_transform into a seeded apply_token_fn.

    Seeds python/numpy/torch RNGs so crop/noise/color_jitter are reproducible,
    and converts to RGB (training does .convert('RGB') after the sequence).
    """
    def apply(image, token, seed):
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        return transformer.apply_transform(image, token).convert("RGB")
    return apply


def make_recall_aug_apply_fn(
    color_jitter_params=(0.3, 0.3, 0.3, 0.3),   # brightness, contrast, saturation, hue
    max_crop_frac: float = 0.15,
    jpeg_quality=(20, 70),
    noise_sigma=(5, 20),   # not part of the recall test; matches training transformer
):
    """Atomic-token apply_fn that reproduces the augmentations from your recall test
    (the `_augmentations` method). Same params: symmetric crop up to max_crop_frac per
    side, ColorJitter(0.3,0.3,0.3,0.3), JPEG quality 20-70, rotate(..., expand=True),
    vertical_flip = ImageOps.flip, horizontal_flip = ImageOps.mirror, grayscale = L->RGB.

    Tokens not present in the recall set (noop, rotate_270, horizontal_flip,
    noise_adding) use the consistent atomic equivalent.
    """
    from PIL import ImageOps
    import torchvision.transforms as transforms

    def apply(image, token, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        w, h = image.size
        cj = transforms.ColorJitter(*color_jitter_params)

        if token == "noop":
            return image.copy()
        if token == "grayscale":
            return image.convert("L").convert("RGB")
        if token == "rotate_90":
            return image.rotate(90, expand=True).convert("RGB")
        if token == "rotate_180":
            return image.rotate(180, expand=True).convert("RGB")
        if token == "rotate_270":
            return image.rotate(270, expand=True).convert("RGB")
        if token == "horizontal_flip":
            return ImageOps.mirror(image).convert("RGB")
        if token == "vertical_flip":
            return ImageOps.flip(image).convert("RGB")
        if token == "color_jitter":
            return cj(image).convert("RGB")
        if token == "crop":
            dw = int(w * max_crop_frac * rng.rand())
            dh = int(h * max_crop_frac * rng.rand())
            left, upper, right, lower = dw, dh, w - dw, h - dh
            if right > left and lower > upper:
                return image.crop((left, upper, right, lower)).convert("RGB")
            return image.convert("RGB")
        if token == "noise_adding":
            arr = np.array(image.convert("RGB")).astype(np.float32)
            arr += rng.normal(0, random.uniform(*noise_sigma), arr.shape)
            return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
        if token == "jpeg_artefacts":
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=random.randint(*jpeg_quality), optimize=True)
            buf.seek(0)
            return Image.open(buf).convert("RGB")
        raise ValueError(f"Unknown token: {token}")
    return apply


class TokenConfusionBenchmark:
    def __init__(
        self,
        model,
        dataset_root: str,
        tokenizer,
        preprocess: Callable[[Image.Image], torch.Tensor],
        device: torch.device,
        n_samples: int = 5000,
        exclude_domains=("quickdraw",),
        max_gen_len: int = 15,
        val_size: float = 0.1,
        random_seed: int = 42,
        seed: int = 2026,
        apply_token_fn: Optional[Callable[[Image.Image, str, int], Image.Image]] = None,
        tokens: Optional[list] = None,   # default: ATOMIC_TOKENS; pass transformer.transformations
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.device = device
        self.n_samples = n_samples
        self.exclude_domains = set(exclude_domains)
        self.max_gen_len = max_gen_len
        self.seed = seed
        # default to the recall-test augmentations; pass apply_token_fn only to override
        self.apply_token_fn = apply_token_fn if apply_token_fn is not None else make_recall_aug_apply_fn()
        self.tokens = tokens if tokens is not None else ATOMIC_TOKENS

        self.dataset = SimpleDomainNetDataset(
            data_dir=dataset_root, split="val",
            val_size=val_size, random_seed=random_seed,
        )

    @staticmethod
    def _seed(image, token):
        return int(hashlib.sha256(image.tobytes() + token.encode()).hexdigest(), 16) % (2 ** 32)

    @staticmethod
    def _is_grayscale(image: Image.Image) -> bool:
        """Same logic as ImageTransformer.is_grayscale: catches L/LA/P modes and
        grayscale images stored as 3-channel RGB (all channels equal)."""
        if image.mode in ("L", "LA", "P"):
            return True
        arr = np.array(image)
        if len(arr.shape) == 3 and arr.shape[2] == 3:
            return np.allclose(arr[:, :, 0], arr[:, :, 1]) and \
                np.allclose(arr[:, :, 1], arr[:, :, 2])
        return False

    def _apply(self, image, token, seed):
        return self.apply_token_fn(image, token, seed)

    @torch.no_grad()
    def _predict(self, orig, transformed):
        o = self.preprocess(orig).unsqueeze(0).to(self.device)
        t = self.preprocess(transformed).unsqueeze(0).to(self.device)
        ids = self.model.generate(
            image_batch_1=o, image_batch_2=t,
            max_new_tokens=self.max_gen_len - 1, do_sample=False,
        ).squeeze(0)
        toks = self.tokenizer.decode(ids, skip_special_tokens=True)
        return toks[0] if len(toks) > 0 else "empty"

    def run(self, model_name, output_path, verbose=True):
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        counts = defaultdict(lambda: defaultdict(int))

        idxs = list(range(len(self.dataset)))
        random.shuffle(idxs)

        self.model.eval()
        done = 0
        skipped_gray = 0
        pbar = tqdm(total=self.n_samples, desc=f"{model_name} confusion")
        for i in idxs:
            if done >= self.n_samples:
                break
            try:
                orig_pil, domain = self.dataset[i]
                if domain in self.exclude_domains:
                    continue
                orig = orig_pil.convert("RGB")
            except Exception:
                continue
            # only keep COLOR images: grayscale->noop would otherwise pollute the matrix
            if self._is_grayscale(orig):
                skipped_gray += 1
                continue
            for tok in self.tokens:
                try:
                    trans = self._apply(orig, tok, self._seed(orig, tok))
                    pred = self._predict(orig, trans)
                except Exception:
                    continue
                counts[tok][pred] += 1
            done += 1
            pbar.update(1)
        pbar.close()

        # build label order: targets as rows, observed preds as cols
        rows = list(self.tokens)
        seen = {p for r in counts for p in counts[r]}
        cols = [c for c in self.tokens if c in seen]
        cols += sorted(seen - set(self.tokens) - {"empty"})
        if "empty" in seen:
            cols.append("empty")

        results = {
            "row_labels": rows,
            "col_labels": cols,
            "counts": {r: dict(counts[r]) for r in rows},
        }
        metadata = {
            "n_images": done,
            "n_pairs": done * len(self.tokens),
            "skipped_grayscale_images": skipped_gray,
            "atomic_tokens": list(self.tokens),
            "excluded_domains": sorted(self.exclude_domains),
            "seed": self.seed,
            "prediction": "first output token, or 'empty' if no match",
        }

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        if out.exists():
            try:
                data = json.loads(out.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        data[model_name] = {"metadata": metadata, "results": results}
        out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

        if verbose:
            print(f"\n[{model_name}] {done} images / {done * len(self.tokens)} pairs -> {output_path}")
        return results


def plot_token_confusion(json_path, model_name, mode="percent", decimals=1,
                         save_path=None, figsize=(9, 7), cmap="Blues"):
    """Paper-style confusion heatmap for a single model.

    mode    : "percent" (row-normalized %, default) or "count" (raw counts)
    decimals: digits after the dot for the % annotations (1 or 2)
    Zero cells are left blank so the diagonal and the real leakage stand out.
    """
    import matplotlib.pyplot as plt

    res = json.loads(Path(json_path).read_text(encoding="utf-8"))[model_name]["results"]
    rows, cols, counts = res["row_labels"], res["col_labels"], res["counts"]
    M = np.array([[counts.get(r, {}).get(c, 0) for c in cols] for r in rows], dtype=float)

    if mode == "percent":
        s = M.sum(1, keepdims=True); s[s == 0] = 1.0
        shown = M / s * 100.0
        vmax = 100.0
        def annot(v): return f"{v:.{decimals}f}" if v > 0 else ""
        cbar_label = "%"
    else:
        shown = M
        vmax = M.max() if M.size else 1.0
        def annot(v): return f"{int(v)}" if v > 0 else ""
        cbar_label = "count"

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(shown, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(model_name)

    thr = vmax * 0.6
    for r in range(len(rows)):
        for c in range(len(cols)):
            txt = annot(shown[r, c])
            if txt:
                ax.text(c, r, txt, ha="center", va="center", fontsize=7,
                        color="white" if shown[r, c] > thr else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, ax