import os
from typing import Dict, List, Tuple, Callable, Optional
from pathlib import Path
import random
import torch
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
import io
from tqdm import tqdm
import json


class DomainRecallEvaluator:
    def __init__(
        self,
        data_dir: str,
        sim_fn: Callable[[Image.Image, Image.Image], float],
        preprocess: Optional[Callable] = None,
        seed: int = 2025,
        val_size: float = 0.1,
        split_random_seed: int = 42
    ):
        self.data_dir = Path(data_dir)
        self.sim_fn = sim_fn
        self.preprocess = preprocess
        self.seed = seed
        self.val_size = val_size
        self.split_random_seed = split_random_seed
        self._set_seed()
        self.domain_to_val_paths: Dict[str, List[Path]] = {}
        self._load_val_split()

    def _set_seed(self) -> None:
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def _load_val_split(self) -> None:
        rng = random.Random(self.split_random_seed)
        for domain in sorted(os.listdir(self.data_dir)):
            domain_path = self.data_dir / domain
            if not domain_path.is_dir():
                continue
            all_paths = []
            for file in sorted(domain_path.iterdir()):
                if file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    all_paths.append(file)
            if not all_paths:
                continue
            shuffled = rng.sample(all_paths, len(all_paths))
            n_val = int(len(shuffled) * self.val_size)
            val_paths = shuffled[:n_val]
            if val_paths:
                self.domain_to_val_paths[domain] = val_paths

    @staticmethod
    def _augmentations(img: Image.Image) -> List[Tuple[str, Image.Image]]:
        augs = []
        w, h = img.size
        img_hash_seed = hash(img.tobytes()) % (2**32)
        random.seed(img_hash_seed)
        torch.manual_seed(img_hash_seed)
        rng = np.random.RandomState(img_hash_seed)
        augs.append(("rotate_90", img.rotate(90, expand=True)))
        augs.append(("rotate_180", img.rotate(180, expand=True)))
        augs.append(("flip_vertical", ImageOps.flip(img)))
        augs.append(("grayscale", img.convert("L").convert("RGB")))
        color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)
        augs.append(("color_jitter", color_jitter(img)))
        max_crop_frac = 0.15
        dw = int(w * max_crop_frac * rng.rand())
        dh = int(h * max_crop_frac * rng.rand())
        left, upper = dw, dh
        right, lower = w - dw, h - dh
        if right > left and lower > upper:
            cropped = img.crop((left, upper, right, lower))
            augs.append(("crop", cropped))
        buffer = io.BytesIO()
        quality = random.randint(20, 70)
        img.save(buffer, format='JPEG', quality=quality, optimize=True)
        buffer.seek(0)
        jpeg_img = Image.open(buffer).convert('RGB')
        augs.append(("jpeg_artefacts", jpeg_img))
        dw2 = int(w * max_crop_frac * rng.rand())
        dh2 = int(h * max_crop_frac * rng.rand())
        left2, upper2 = dw2, dh2
        right2, lower2 = w - dw2, h - dh2
        if right2 > left2 and lower2 > upper2:
            cropped2 = img.crop((left2, upper2, right2, lower2))
            cropped_jittered = color_jitter(cropped2)
            cropped_jittered_flipped = ImageOps.flip(cropped_jittered)
            augs.append(("crop_colorjitter_flipv", cropped_jittered_flipped))
        flipped = ImageOps.mirror(img)
        rotated = flipped.rotate(270, expand=True)
        gray_rgb = rotated.convert("L").convert("RGB")
        augs.append(("fliph_rot270_grayscale", gray_rgb))
        return augs

    def evaluate(
        self,
        n_samples: int = 10,
        threshold: float = 0.5,
        model_name: str = "unknown_model",
        output_path: str = "domain_recall.json",
        verbose: bool = True
    ) -> Dict[str, Dict[str, float]]:
        self._set_seed()
        results: Dict[str, Dict[str, float]] = {}
        for domain in sorted(self.domain_to_val_paths.keys()):
            paths = self.domain_to_val_paths[domain]
            if not paths:
                continue
            n = min(n_samples, len(paths))
            sampled_paths = random.sample(paths, n)
            aug_scores: Dict[str, List[float]] = {}
            for p in tqdm(sampled_paths, desc=f"Recall: {domain}", leave=False):
                orig = None
                try:
                    orig = Image.open(p).convert("RGB")
                    aug_list = self._augmentations(orig)
                    for aug_name, aug_img in aug_list:
                        score = 0.0
                        try:
                            with torch.inference_mode():
                                if self.preprocess is not None:
                                    orig_proc = self.preprocess(orig)
                                    aug_proc = self.preprocess(aug_img)
                                    raw_score = self.sim_fn(orig_proc, aug_proc)
                                else:
                                    raw_score = self.sim_fn(orig, aug_img)
                            if torch.is_tensor(raw_score):
                                score = raw_score.detach().cpu().item()
                            else:
                                score = float(raw_score)
                        except Exception:
                            score = 0.0
                        aug_scores.setdefault(aug_name, []).append(score)
                        del aug_img
                    del orig
                except Exception as e:
                    if orig is not None:
                        del orig
                    continue
            recall_per_aug = {
                aug_name: sum(s >= threshold for s in scores) / len(scores) if scores else 0.0
                for aug_name, scores in aug_scores.items()
            }
            results[domain] = recall_per_aug
            if verbose:
                print(f"Domain: {domain} | N = {n}")
                for aug, rec in sorted(recall_per_aug.items()):
                    print(f"  {aug:15s} --> recall = {rec:.3f}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if Path(output_path).exists():
            with open(output_path, "r", encoding="utf-8") as f:
                try:
                    all_data = json.load(f)
                except Exception:
                    all_data = {}
        else:
            all_data = {}
        metadata = {
            "n_samples": n_samples,
            "threshold": threshold,
            "seed": self.seed,
            "val_size": self.val_size,
            "split_random_seed": self.split_random_seed
        }
        all_data[model_name] = {"metadata": metadata, "results": results}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"\nSaved domain recall for '{model_name}' to '{output_path}'")
        return results

class DomainFprEvaluator:
    def __init__(
        self,
        data_dir: str,
        sim_fn: Callable,
        preprocess: Optional[Callable] = None,
        seed: int = 2025,
        val_size: float = 0.1,
        split_random_seed: int = 42
    ):
        self.data_dir = Path(data_dir)
        self.sim_fn = sim_fn
        self.preprocess = preprocess
        self.seed = seed
        self.val_size = val_size
        self.split_random_seed = split_random_seed
        self._set_seed()
        self.domain_to_val_paths: Dict[str, List[Path]] = {}
        self._load_val_split()

    def _set_seed(self) -> None:
        random.seed(self.seed)
        torch.manual_seed(self.seed)

    def _load_val_split(self) -> None:
        rng = random.Random(self.split_random_seed)
        for domain in sorted(os.listdir(self.data_dir)):
            domain_path = self.data_dir / domain
            if not domain_path.is_dir():
                continue
            all_paths = []
            for file in sorted(domain_path.iterdir()):
                if file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    all_paths.append(file)
            if not all_paths:
                continue
            shuffled = rng.sample(all_paths, len(all_paths))
            n_val = int(len(shuffled) * self.val_size)
            val_paths = shuffled[:n_val]
            if len(val_paths) >= 2:
                self.domain_to_val_paths[domain] = val_paths

    def _generate_pairs_deterministic(self, paths: List[Path], n_samples: int) -> List[Tuple[Path, Path]]:
        if len(paths) < 2:
            return []
        all_pairs = []
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                all_pairs.append((paths[i], paths[j]))
        if not all_pairs:
            return []
        domain_seed = hash(tuple(sorted(p.name for p in paths))) % (2**32)
        pair_rng = random.Random(domain_seed)
        pair_rng.shuffle(all_pairs)
        return all_pairs[:n_samples]

    def evaluate(
        self,
        n_samples: int = 10,
        threshold: float = 0.5,
        model_name: str = "unknown_model",
        output_path: str = "domain_fpr.json",
        verbose: bool = True
    ) -> Dict[str, float]:
        self._set_seed()
        results: Dict[str, float] = {}
        for domain in tqdm(sorted(self.domain_to_val_paths.keys()), desc="Domains"):
            paths = self.domain_to_val_paths[domain]
            if len(paths) < 2:
                results[domain] = 1.0
                continue
            pairs = self._generate_pairs_deterministic(paths, n_samples)
            if not pairs:
                results[domain] = 1.0
                continue
            fp = 0
            for p1, p2 in pairs:
                img1, img2 = None, None
                try:
                    img1 = Image.open(p1).convert("RGB")
                    img2 = Image.open(p2).convert("RGB")
                    score = 1.0
                    try:
                        if self.preprocess is not None:
                            x1 = self.preprocess(img1)
                            x2 = self.preprocess(img2)
                            raw_score = self.sim_fn(x1, x2)
                        else:
                            raw_score = self.sim_fn(img1, img2)
                        if torch.is_tensor(raw_score):
                            score = raw_score.detach().cpu().item()
                        else:
                            score = float(raw_score)
                    except Exception:
                        score = 1.0
                    if score >= threshold:
                        fp += 1
                except Exception:
                    fp += 1
                finally:
                    if img1 is not None:
                        del img1
                    if img2 is not None:
                        del img2
            fpr = fp / len(pairs)
            results[domain] = fpr
            if verbose:
                print(f"Domain: {domain} | N = {len(pairs)} | FPR = {fp}/{len(pairs)} = {fpr:.3f}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if Path(output_path).exists():
            with open(output_path, "r", encoding="utf-8") as f:
                try:
                    all_data = json.load(f)
                except Exception:
                    all_data = {}
        else:
            all_data = {}
        metadata = {
            "n_samples": n_samples,
            "threshold": threshold,
            "seed": self.seed,
            "val_size": self.val_size,
            "split_random_seed": self.split_random_seed,
            "metric": "FPR on intra-domain val negative pairs"
        }
        all_data[model_name] = {"metadata": metadata, "results": results}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"\nSaved domain FPR for '{model_name}' to '{output_path}'")
        return results