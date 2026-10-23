#!/usr/bin/env python3
"""
Pretraining entry point for the v2 vocabulary (jpeg + fragment edits).

Identical to ``plagiarism.run_train_plagiarism`` except for the data side:
``TransformTokenizerV2`` / ``ImageTransformerV2`` replace the v1 tokenizer and
transformer. The model class, the DomainNet loaders and the training loop are
reused unchanged. The model is trained from scratch (vocab_size 16).

Run from the repository root:
    python -m plagiarism_v2.run_train_plagiarism_v2 \
        --config plagiarism_v2/configs/train_config_plagiarism_v2.yaml \
        --data_path data
"""

import argparse

import torch
from omegaconf import OmegaConf

from src.dataset import AugmentationScheduler, get_domainnet_dataloaders
from plagiarism.effnet_plagiarism import ImageTransformPlagiarismPredictor
from plagiarism.train_plagiarism import train_model

from plagiarism_v2.dataset_v2 import ImageTransformerV2, TransformTokenizerV2, VOCAB_SIZE_V2


def _float_dict(d) -> dict:
    return {str(k): float(v) for k, v in dict(d or {}).items()}


def _groups(v):
    """YAML list of lists -> list of lists of str, or None to use the defaults."""
    if v is None:
        return None
    return [[str(t) for t in g] for g in v]


def build_transformer(cfg) -> ImageTransformerV2:
    """Construct ImageTransformerV2 from the optional ``augmentation_v2`` config block."""
    a = cfg.get("augmentation_v2", {}) or {}
    text_langs = a.get("text_langs", None)
    return ImageTransformerV2(
        exclusive_groups=_groups(a.get("exclusive_groups", None)),
        sticker_dir=a.get("sticker_dir", None),
        sticker_asset_prob=float(a.get("sticker_asset_prob", 0.5)),
        font_dirs=list(a.get("font_dirs", []) or []),
        text_corpus_dir=a.get("text_corpus_dir", None),
        text_langs=list(text_langs) if text_langs else None,
        text_lang_weights=_float_dict(a.get("text_lang_weights", {})),
        text_layout_weights=_float_dict(a.get("text_layout_weights", {})),
        text_min_net_px=float(a.get("text_min_net_px", 9.0)),
        fragment_kinds=_float_dict(a.get("fragment_kinds", {})) or None,
        region_shapes=_float_dict(a.get("region_shapes", {})) or None,
        n_fragment_edits_probs=list(a.get("n_fragment_edits_probs", [0.7, 0.3])),
        crop_pad_prob=float(a.get("crop_pad_prob", 0.3)),
        crop_pad_styles=_float_dict(a.get("crop_pad_styles", {})) or None,
        visibility_threshold=float(a.get("visibility_threshold", 0.6)),
        visibility_min_frac=float(a.get("visibility_min_frac", 0.006)),
        visibility_retries=int(a.get("visibility_retries", 2)),
        token_p_scale=_float_dict(a.get("token_p_scale", {})),
    )


def main(config_path: str, data_path: str) -> None:
    cfg = OmegaConf.load(config_path)

    if int(cfg.model.decoder.vocab_size) != VOCAB_SIZE_V2:
        raise ValueError(
            f"model.decoder.vocab_size must be {VOCAB_SIZE_V2} for the v2 vocabulary, "
            f"got {cfg.model.decoder.vocab_size}"
        )

    # --- Model (unchanged class) ---
    model = ImageTransformPlagiarismPredictor(cfg.model)

    init_weights = cfg.model.get("initialization_weights_path")
    if init_weights is not None:  # same plain loading as in v1 (must be a v2 checkpoint)
        print(f"[init] loading pretrained weights: {init_weights}")
        state_dict = torch.load(init_weights, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[init] missing keys: {len(missing)}")
        print(f"[init] unexpected keys: {len(unexpected)}")

    # --- Data pipeline (v2 vocabulary) ---
    tokenizer = TransformTokenizerV2()
    transformer = build_transformer(cfg)
    print(f"[aug-v2] tokens: {transformer.transformations}")
    print(f"[aug-v2] fragment kinds: {dict(transformer.fragment_editor.kind_weights)}")
    print(f"[aug-v2] exclusive groups: {[sorted(g) for g in transformer.exclusive_groups]}")
    print(f"[aug-v2] text languages: {transformer.text_langs}")
    print(f"[aug-v2] truetype fonts found: {len(transformer.font_bank.paths)} | "
          f"sticker assets: {len(transformer.fragment_editor.sticker_bank.paths)}")

    aug_config = cfg.get("augmentation", {})
    augmentation_scheduler = AugmentationScheduler(
        initial_p=aug_config.get("initial_p", 0.3),
        milestones=list(aug_config.get("milestones", [])),
        probs=list(aug_config.get("probs", [])),
    )

    dataloaders = get_domainnet_dataloaders(
        data_path,
        tokenizer,
        transformer,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.get("num_workers", 4),
        val_size=cfg.training.get("val_size", 0.1),
        random_seed=cfg.training.get("random_seed", 42),
        max_seq_len=cfg.model.decoder.max_seq_len,
        augmentation_scheduler=augmentation_scheduler,
    )

    # --- Training loop (unchanged) ---
    train_model(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        config=cfg,
        augmentation_scheduler=augmentation_scheduler,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pretrain ImageTransformPlagiarismPredictor with the v2 augmentation vocabulary."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="plagiarism_v2/configs/train_config_plagiarism_v2.yaml",
        help="Path to OmegaConf yaml.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data",
        help="Root directory with <domain>/*.{jpg,png} subfolders.",
    )
    args = parser.parse_args()
    main(args.config, args.data_path)
