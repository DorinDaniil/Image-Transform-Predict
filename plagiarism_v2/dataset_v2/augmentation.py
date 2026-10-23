"""
ImageTransformerV2 -- v1 augmentations + jpeg_artefacts + text_overlay + fragment_edit.

Public API is the same as ``src.dataset.ImageTransformer`` so that the existing
datasets (``DomainNetDataset``, ``CleanNegativePairsDataset``, benchmarks) can
consume it without changes:

    transform(image, p)            -> (image, [token, ...])
    transform_by_length(image, k)  -> (image, [token, ...])   (benchmarks)
    set_p(p)

Sampling keeps the v1 semantics (independent Bernoulli(p) per token with the
v1 geometric / colour constraints) and adds the v2 compatibility rules below.

Compatibility rules
-------------------
``exclusive_groups`` -- at most ONE token from each group per sequence
(configurable, defaults):
    * {jpeg_artefacts, noise_adding}   two heavy degradations hide each other
    * {text_overlay, fragment_edit}    one local edit per pair
Application order is random (v1 semantics): text may get flipped, rotated or
cropped by a later token, exactly like any other content.
``crop`` (id 10, as in v1) means "field of view changed": with probability
``crop_pad_prob`` margins are ADDED (frame / document margins / letterbox,
see ``reframe.py``) instead of cut away. A ``text_overlay`` in the
``caption_outside`` layout also extends the canvas, so it is labelled
``text_overlay`` + ``crop``.
Observability -- every text_overlay / fragment_edit that leaves almost no trace
  at network resolution is retried and, failing that, dropped from the label.
  "Visible" = mean |diff| >= visibility_threshold OR the fraction of pixels whose
  luminance changed by > 12 >= visibility_min_frac (so a small but crisp
  watermark counts, while a faint global haze does not).

After each ``transform`` call ``last_fragment_kinds`` / ``last_text_detail``
describe what was actually applied (for per-kind analysis and previews).
"""

import random
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

from src.dataset.augmentation import ImageTransformer

from .overlays import FRAGMENT_KINDS, FontBank, FragmentEditor, change_stats
from .reframe import apply_pad
from .tokenizer import TRANSFORM_TOKENS_V2

_SPECIAL = {"[PAD]", "[START]", "[END]"}

DEFAULT_EXCLUSIVE_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("jpeg_artefacts", "noise_adding"),
    ("text_overlay", "fragment_edit"),
)


class ImageTransformerV2(ImageTransformer):
    TEXT_TOKEN = "text_overlay"
    FRAGMENT_TOKEN = "fragment_edit"
    LOCAL_TOKENS = (TEXT_TOKEN, FRAGMENT_TOKEN)

    def __init__(
        self,
        sticker_dir: Optional[str] = None,
        sticker_asset_prob: float = 0.5,
        font_dirs: Optional[Sequence[str]] = None,
        text_corpus_dir: Optional[str] = None,
        text_langs: Optional[Sequence[str]] = None,
        text_lang_weights: Optional[Dict[str, float]] = None,
        text_layout_weights: Optional[Dict[str, float]] = None,
        text_min_net_px: float = 9.0,
        fragment_kinds: Optional[Dict[str, float]] = None,
        region_shapes: Optional[Dict[str, float]] = None,
        n_fragment_edits_probs: Sequence[float] = (0.7, 0.3),
        exclusive_groups: Optional[Sequence[Sequence[str]]] = None,
        crop_pad_prob: float = 0.3,
        crop_pad_styles: Optional[Dict[str, float]] = None,
        visibility_threshold: float = 0.6,
        visibility_min_frac: float = 0.006,
        visibility_retries: int = 2,
        token_p_scale: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.transform_tokens = {k: v for k, v in TRANSFORM_TOKENS_V2.items() if k not in _SPECIAL}
        self.transformations = list(self.transform_tokens.keys())

        self.visibility_threshold = float(visibility_threshold)
        self.visibility_min_frac = float(visibility_min_frac)
        self.visibility_retries = int(visibility_retries)
        self.crop_pad_prob = float(crop_pad_prob)
        self.crop_pad_styles = dict(crop_pad_styles or {})
        self.token_p_scale = dict(token_p_scale or {})
        self.n_fragment_edits_probs = [float(p) for p in n_fragment_edits_probs]
        if not self.n_fragment_edits_probs or sum(self.n_fragment_edits_probs) <= 0:
            raise ValueError("n_fragment_edits_probs must be a non-empty list of weights for 1, 2, ... edits")

        groups = exclusive_groups if exclusive_groups is not None else DEFAULT_EXCLUSIVE_GROUPS
        self.exclusive_groups: List[frozenset] = []
        for g in groups:
            g = [str(t) for t in g]
            unknown = [t for t in g if t not in self.transform_tokens]
            if unknown:
                raise ValueError(f"exclusive_groups: unknown tokens {unknown}")
            self.exclusive_groups.append(frozenset(g))

        self.font_bank = FontBank(font_dirs)
        self.fragment_editor = FragmentEditor(
            font_bank=self.font_bank,
            sticker_dir=sticker_dir,
            sticker_asset_prob=sticker_asset_prob,
            text_corpus_dir=text_corpus_dir,
            text_langs=text_langs,
            text_lang_weights=text_lang_weights,
            text_layout_weights=text_layout_weights,
            text_min_net_px=text_min_net_px,
            kind_weights=fragment_kinds,
            region_shapes=region_shapes,
        )
        self.last_fragment_kinds: List[str] = []
        self.last_fragment_details: List[Optional[str]] = []
        self.last_text_detail: Optional[str] = None  # "layout/lang" of the last text_overlay

    # ------------------------------------------------------------------ info
    @property
    def fragment_kinds(self) -> Tuple[str, ...]:
        return FRAGMENT_KINDS

    @property
    def text_langs(self) -> List[str]:
        return self.fragment_editor.text_bank.langs

    # ------------------------------------------------------------------ ops
    def apply_transform(self, image: Image.Image, transform: str) -> Image.Image:
        if transform == self.TEXT_TOKEN:
            out, ok = self._apply_text_overlay(image)
            return out
        if transform == self.FRAGMENT_TOKEN:
            out, ok = self._apply_fragment_edit(image)
            return out
        if transform == "crop" and random.random() < self.crop_pad_prob:
            return apply_pad(image, self.crop_pad_styles)  # field of view extended instead of cut
        return super().apply_transform(image, transform)

    def _visible(self, before: Image.Image, after: Image.Image) -> bool:
        if self.visibility_threshold <= 0 and self.visibility_min_frac <= 0:
            return True
        mean, frac = change_stats(before, after)
        return (self.visibility_threshold > 0 and mean >= self.visibility_threshold) or \
               (self.visibility_min_frac > 0 and frac >= self.visibility_min_frac)

    def _apply_text_overlay(self, image: Image.Image, layout: Optional[str] = None) -> Tuple[Image.Image, bool]:
        """Apply one text overlay; retried while invisible. Returns (image, visible)."""
        for _ in range(self.visibility_retries + 1):
            cand, detail = self.fragment_editor.apply_text(image, layout)
            if self._visible(image, cand):
                self.last_text_detail = detail
                return cand, True
        self.last_text_detail = None
        return image, False

    def _sample_fragment_kinds(self) -> List[str]:
        """1..n distinct kinds, weighted."""
        n = random.choices(range(1, len(self.n_fragment_edits_probs) + 1), weights=self.n_fragment_edits_probs)[0]
        kinds: List[str] = []
        for _ in range(n):
            kinds.append(self.fragment_editor.sample_kind(exclude=kinds))
        return kinds

    def _apply_fragment_edit(self, image: Image.Image, kinds: Optional[Sequence[str]] = None) -> Tuple[Image.Image, bool]:
        """Apply 1..n local edits of distinct kinds; returns (image, at_least_one_visible)."""
        if kinds is None:
            kinds = self._sample_fragment_kinds()
        out = image
        applied: List[str] = []
        details: List[Optional[str]] = []
        for kind in kinds:
            for _ in range(self.visibility_retries + 1):
                cand, _k = self.fragment_editor.apply(out, kind)
                if self._visible(out, cand):
                    out = cand
                    applied.append(kind)
                    details.append(self.fragment_editor.last_detail)
                    break
        self.last_fragment_kinds = applied
        self.last_fragment_details = details
        return out, bool(applied)

    def _apply_checked(self, image: Image.Image, transform: str) -> Tuple[Image.Image, bool]:
        if transform == self.TEXT_TOKEN:
            return self._apply_text_overlay(image)
        if transform == self.FRAGMENT_TOKEN:
            return self._apply_fragment_edit(image)
        return self.apply_transform(image, transform), True

    def _apply_sequence(self, image: Image.Image, sequence: List[str]) -> Tuple[Image.Image, List[str]]:
        """Apply tokens in the sampled (random) order; label = tokens that were actually applied."""
        self.last_fragment_kinds, self.last_fragment_details = [], []
        self.last_text_detail = None
        out = image.copy()
        applied: List[str] = []
        for t in sequence:
            before_size = out.size
            out, ok = self._apply_checked(out, t)
            if ok:
                applied.append(t)
                # a caption strip added under / above the picture extends the canvas:
                # that is a change of the field of view, i.e. also a ``crop``
                if t == self.TEXT_TOKEN and out.size != before_size:
                    applied.append("crop")
        # keep each token once (crop may have been added by caption_outside and sampled too)
        seen, unique = set(), []
        for t in applied:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        applied = unique
        if not applied:
            applied = ["noop"]
        return out.convert("RGB"), applied

    def transform(self, image: Image.Image, p: Optional[float] = None) -> Tuple[Image.Image, List[str]]:
        actual_p = p if p is not None else self._current_p
        sequence = self.sample_transformations(image, actual_p)
        return self._apply_sequence(image, sequence)

    def transform_by_length(self, image: Image.Image, k: int) -> Tuple[Image.Image, List[str]]:
        sequence = self.sample_transformations_by_length(image, k)
        return self._apply_sequence(image, sequence)

    # -------------------------------------------------------------- sampler
    @staticmethod
    def _new_flags(grayscale_selected: bool) -> Dict[str, object]:
        return {
            "r180": False, "h": False, "v": False, "rot": False,
            "grayscale_selected": grayscale_selected,
        }

    def _is_allowed(self, t: str, flags: Dict[str, object], is_gray: bool, selected: List[str]) -> bool:
        if t in selected:
            return False
        for g in self.exclusive_groups:  # jpeg vs noise, text vs fragment, ...
            if t in g and any(s in g for s in selected):
                return False
        if t == "grayscale":
            return not is_gray and not flags["grayscale_selected"]
        if t == "color_jitter":
            return not flags["grayscale_selected"]
        if t in ("rotate_90", "rotate_270"):
            return not flags["rot"]
        if t == "rotate_180":
            return not flags["rot"] and not (flags["h"] and flags["v"])
        if t == "horizontal_flip":
            return not (flags["r180"] and flags["v"])
        if t == "vertical_flip":
            return not (flags["r180"] and flags["h"])
        return True  # crop, jpeg_artefacts, noise_adding, text_overlay, fragment_edit

    @staticmethod
    def _mark(t: str, flags: Dict[str, object]) -> None:
        if t == "grayscale":
            flags["grayscale_selected"] = True
        elif t in ("rotate_90", "rotate_270"):
            flags["rot"] = True
        elif t == "rotate_180":
            flags["r180"] = True
            flags["rot"] = True
        elif t == "horizontal_flip":
            flags["h"] = True
        elif t == "vertical_flip":
            flags["v"] = True

    def sample_transformations(self, image: Image.Image, p: float = 0.3) -> List[str]:
        is_gray = self.is_grayscale(image)
        candidates = [t for t in self.transformations if t != "noop"]
        random.shuffle(candidates)

        selected: List[str] = []
        flags = self._new_flags(grayscale_selected=False)  # v1 semantics
        for t in candidates:
            if not self._is_allowed(t, flags, is_gray, selected):
                continue
            if random.random() >= p * self.token_p_scale.get(t, 1.0):
                continue
            selected.append(t)
            self._mark(t, flags)
        return selected if selected else ["noop"]

    def sample_transformations_by_length(self, image: Image.Image, k: int) -> List[str]:
        if not (1 <= k <= 5):
            raise ValueError("k must be between 1 and 5")
        is_gray = self.is_grayscale(image)
        candidates = [t for t in self.transformations if t != "noop"]
        random.shuffle(candidates)

        if k == 1 and random.random() < 0.05:
            return ["noop"]

        selected: List[str] = []
        flags = self._new_flags(grayscale_selected=is_gray)
        for t in candidates:
            if len(selected) >= k:
                break
            if self._is_allowed(t, flags, is_gray, selected):
                selected.append(t)
                self._mark(t, flags)

        while len(selected) < k:
            possible = [t for t in candidates if self._is_allowed(t, flags, is_gray, selected)]
            if not possible:
                break
            t = random.choice(possible)
            selected.append(t)
            self._mark(t, flags)

        while len(selected) < k:  # v1 fallback padding
            selected.append("color_jitter")
        return selected[:k]
