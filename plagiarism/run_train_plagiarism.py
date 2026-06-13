#!/usr/bin/env python3
"""
Pretraining entry point for ImageTransformPlagiarismPredictor.

Run from the repository root:
    python -m plagiarism.run_train_plagiarism \
        --config plagiarism/configs/train_config_plagiarism.yaml \
        --data_path data
"""

import argparse

import torch
from omegaconf import OmegaConf

# Shared data pipeline lives in the main package.
from src.dataset import (
    AugmentationScheduler,
    ImageTransformer,
    TransformTokenizer,
    get_domainnet_dataloaders,
)
from plagiarism.effnet_plagiarism import ImageTransformPlagiarismPredictor
from plagiarism.train_plagiarism import train_model


def main(config_path: str, data_path: str) -> None:
    cfg = OmegaConf.load(config_path)

    # --- Model ---
    model = ImageTransformPlagiarismPredictor(cfg.model)

    init_weights = cfg.model.get("initialization_weights_path")
    if init_weights is not None:
        print(f"[init] loading pretrained weights: {init_weights}")
        state_dict = torch.load(init_weights, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[init] missing keys: {len(missing)}")
        print(f"[init] unexpected keys: {len(unexpected)}")

    # --- Data pipeline ---
    tokenizer = TransformTokenizer()
    transformer = ImageTransformer()

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

    train_model(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        config=cfg,
        augmentation_scheduler=augmentation_scheduler,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pretrain ImageTransformPlagiarismPredictor on DomainNet-like data with BxB random pairs."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="plagiarism/configs/train_config_plagiarism.yaml",
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
