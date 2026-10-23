#!/usr/bin/env python3
"""
Hard-negative tuning entry point for the v2 vocabulary.

Reuses ``CleanNegativePairsDataset`` and the tuning loop from the v1 package
unchanged; only the tokenizer / transformer are the v2 ones.

Run from the repository root:
    python -m plagiarism_v2.run_tune_plagiarism_v2 \
        --config plagiarism_v2/configs/tune_config_plagiarism_v2.yaml \
        --negative-root data/negative_pairs \
        --clean-json data_clean/good_pairs_after_loss_90.json
"""

import argparse

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from plagiarism.effnet_plagiarism import ImageTransformPlagiarismPredictor
from plagiarism.run_tune_plagiarism import CleanNegativePairsDataset
from plagiarism.tuning_plagiarism import train_model

from plagiarism_v2.dataset_v2 import TransformTokenizerV2, VOCAB_SIZE_V2
from plagiarism_v2.run_train_plagiarism_v2 import build_transformer


def main():
    parser = argparse.ArgumentParser(
        description="Tune ImageTransformPlagiarismPredictor (v2 vocabulary) on cleaned hard-negative pairs JSON."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--negative-root", type=str, required=True)
    parser.add_argument("--clean-json", type=str, required=True)
    parser.add_argument("--augmentation-p", type=float, default=0.3)
    parser.add_argument("--val-size", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if int(cfg.model.decoder.vocab_size) != VOCAB_SIZE_V2:
        raise ValueError(f"model.decoder.vocab_size must be {VOCAB_SIZE_V2}, got {cfg.model.decoder.vocab_size}")

    model = ImageTransformPlagiarismPredictor(cfg.model)
    init_weights = cfg.model.get("initialization_weights_path")
    if init_weights is not None:  # same plain loading as in v1 (v2 pretrain checkpoint)
        print(f"[init] loading pretrained weights: {init_weights}")
        state_dict = torch.load(init_weights, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[init] missing keys: {len(missing)}")
        print(f"[init] unexpected keys: {len(unexpected)}")

    tokenizer = TransformTokenizerV2()
    transformer = build_transformer(cfg)
    print(f"[aug-v2] tokens: {transformer.transformations}")
    print(f"[aug-v2] fragment kinds: {dict(transformer.fragment_editor.kind_weights)}")
    print(f"[aug-v2] text languages: {transformer.text_langs}")

    common_kwargs = dict(
        negative_root=args.negative_root,
        pairs_json_path=args.clean_json,
        tokenizer=tokenizer,
        transformer=transformer,
        val_size=args.val_size,
        random_seed=args.seed,
        max_seq_len=cfg.model.decoder.max_seq_len,
        augmentation_p=args.augmentation_p,
    )
    train_ds = CleanNegativePairsDataset(split="train", **common_kwargs)
    val_ds = CleanNegativePairsDataset(split="val", **common_kwargs)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.get("num_workers", 4),
        pin_memory=True,
        drop_last=False,
    )

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        tokenizer=tokenizer,
    )


if __name__ == "__main__":
    main()
