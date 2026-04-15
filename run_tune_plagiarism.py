#!/usr/bin/env python3
"""
Tuning entry point for ImageTransformPlagiarismPredictor on hard negatives.

Usage:
    python run_tune_plagiarism.py \
        --config configs/tune_config_plagiarism.yaml \
        --negative-root /mnt/DATA2/dorin/res.cv.science.dataset.generation/datasets \
        --clean-json data_clean/good_pairs_after_loss_90.json
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.dataset import ImageTransformer, TransformTokenizer
from src.model.effnet_plagiarism import ImageTransformPlagiarismPredictor
from src.tuning_plagiarism import train_model


def _default_image_preprocessor():
    return transforms.Compose(
        [
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _load_pairs_from_json(json_path: Union[str, Path]) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "pairs" in data and isinstance(data["pairs"], list):
            return data["pairs"]
        if "outliers" in data and isinstance(data["outliers"], list):
            return data["outliers"]
        if all(isinstance(v, dict) for v in data.values()):
            return list(data.values())
        raise ValueError(f"Unsupported JSON dict format in {json_path}")
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported JSON format in {json_path}")


def _normalize_pair_item(item: Dict[str, Any]) -> Dict[str, str]:
    if "img1" in item and "img2" in item:
        return {"img1_path": str(item["img1"]), "img2_path": str(item["img2"])}
    if "batch_name" in item and "image_1" in item and "image_2" in item:
        batch_name = str(item["batch_name"])
        return {
            "img1_path": str(Path(batch_name) / "dataset" / str(item["image_1"])),
            "img2_path": str(Path(batch_name) / "dataset" / str(item["image_2"])),
        }
    if "dataset" in item and "image_1" in item and "image_2" in item:
        ds = str(item["dataset"])
        batch_name = f"dataset_{ds}"
        return {
            "img1_path": str(Path(batch_name) / "dataset" / str(item["image_1"])),
            "img2_path": str(Path(batch_name) / "dataset" / str(item["image_2"])),
        }
    raise ValueError(f"Unsupported pair item keys: {sorted(item.keys())}")


class CleanNegativePairsDataset(Dataset):
    """
    Clean negative pairs dataset driven by JSON list.

    Output matches the tuning loop contract:
        (img1, img2, img1_aug, img2_aug, seq1, seq2)
    """

    def __init__(
        self,
        negative_root: str,
        pairs_json_path: str,
        tokenizer: TransformTokenizer,
        transformer: ImageTransformer,
        split: str = "train",
        val_size: float = 0.05,
        random_seed: int = 42,
        max_seq_len: int = 15,
        augmentation_p: float = 0.3,
        image_preprocessor=None,
    ):
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        self.negative_root = Path(negative_root)
        self.tokenizer = tokenizer
        self.transformer = transformer
        self.max_seq_len = max_seq_len
        self.augmentation_p = augmentation_p
        self.preprocessor = image_preprocessor or _default_image_preprocessor()

        raw_items = _load_pairs_from_json(pairs_json_path)
        pairs = [_normalize_pair_item(it) for it in raw_items]

        rng = random.Random(random_seed)
        shuffled = rng.sample(pairs, len(pairs))
        n_val = int(len(shuffled) * val_size)
        self.pairs = shuffled[:n_val] if split == "val" else shuffled[n_val:]
        if not self.pairs:
            raise ValueError("No pairs loaded for the requested split (check val_size and JSON size).")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        rec = self.pairs[idx]
        img1_path = Path(rec["img1_path"])
        img2_path = Path(rec["img2_path"])
        if not img1_path.is_absolute():
            img1_path = self.negative_root / img1_path
        if not img2_path.is_absolute():
            img2_path = self.negative_root / img2_path

        img1_orig = Image.open(img1_path).convert("RGB")
        img2_orig = Image.open(img2_path).convert("RGB")

        img1_aug, seq1 = self.transformer.transform(img1_orig, p=self.augmentation_p)
        img2_aug, seq2 = self.transformer.transform(img2_orig, p=self.augmentation_p)

        img1_t = self.preprocessor(img1_orig)
        img2_t = self.preprocessor(img2_orig)
        img1_aug_t = self.preprocessor(img1_aug.convert("RGB"))
        img2_aug_t = self.preprocessor(img2_aug.convert("RGB"))

        seq1_ids = self.tokenizer.encode(
            transforms=seq1,
            add_special_tokens=True,
            max_seq_len=self.max_seq_len,
            return_targets=False,
        )
        seq2_ids = self.tokenizer.encode(
            transforms=seq2,
            add_special_tokens=True,
            max_seq_len=self.max_seq_len,
            return_targets=False,
        )

        return (
            img1_t,
            img2_t,
            img1_aug_t,
            img2_aug_t,
            seq1_ids.detach().clone().long(),
            seq2_ids.detach().clone().long(),
        )


def main():
    parser = argparse.ArgumentParser(
        description="Tune ImageTransformPlagiarismPredictor on cleaned hard-negative pairs JSON."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--negative-root", type=str, required=True)
    parser.add_argument("--clean-json", type=str, required=True)
    parser.add_argument("--augmentation-p", type=float, default=0.3)
    parser.add_argument("--val-size", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    model = ImageTransformPlagiarismPredictor(cfg.model)
    init_weights = cfg.model.get("initialization_weights_path")
    if init_weights is not None:
        print(f"[init] loading pretrained weights: {init_weights}")
        state_dict = torch.load(init_weights, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[init] missing keys: {len(missing)}")
        print(f"[init] unexpected keys: {len(unexpected)}")

    tokenizer = TransformTokenizer()
    transformer = ImageTransformer()

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
