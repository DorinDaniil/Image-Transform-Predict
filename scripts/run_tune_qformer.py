import argparse
import os
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Union, Optional

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from src.model.model_vit_qformer import ImageTransformPredictorViTQFormer
from src.dataset.tokenizer import TransformTokenizer
from src.dataset.augmentation import ImageTransformer
from src.tuning import train_model



# python run_tune_qformer.py --config configs/train_config_vit_qformer_pretrain.yaml --negative-root "/mnt/DATA2/dorin/res.cv.science.dataset.generation/datasets" --clean-json "data_clean/good_pairs_after_loss_90.json"


def _default_image_preprocessor():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _load_pairs_from_json(json_path: Union[str, Path]) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        # clean_negative_pairs.py: {"threshold": ..., "pairs": [...]}
        if "pairs" in data and isinstance(data["pairs"], list):
            return data["pairs"]
        # other possible: {"threshold": ..., "outliers": [...]}
        if "outliers" in data and isinstance(data["outliers"], list):
            return data["outliers"]
        # dict of id -> obj
        if all(isinstance(v, dict) for v in data.values()):
            return list(data.values())
        raise ValueError(f"Unsupported JSON dict format in {json_path}")

    if isinstance(data, list):
        return data

    raise ValueError(f"Unsupported JSON format in {json_path}")


def _normalize_pair_item(item: Dict[str, Any]) -> Dict[str, str]:
    """
    Normalize different pair formats into:
      {"img1_path": str, "img2_path": str}

    Supported:
      - our cleaned JSON format:
          {"dataset": "...", "img1": "dataset_0/dataset/000..._1.png", "img2": "...", ...}
      - {"batch_name","image_1","image_2", ...}  -> resolved via negative_root/batch_name/dataset/*
      - {"dataset","image_1","image_2", ...}     -> resolved via negative_root/dataset_<dataset>/dataset/*
    """
    # Preferred: paths already present (relative to negative_root, or absolute)
    if "img1" in item and "img2" in item:
        return {"img1_path": str(item["img1"]), "img2_path": str(item["img2"])}

    if "batch_name" in item and "image_1" in item and "image_2" in item:
        batch_name = str(item["batch_name"])
        return {
            "img1_path": str(Path(batch_name) / "dataset" / str(item["image_1"])),
            "img2_path": str(Path(batch_name) / "dataset" / str(item["image_2"])),
        }

    if "dataset" in item and "image_1" in item and "image_2" in item:
        dataset = str(item["dataset"])
        batch_name = f"dataset_{dataset}"
        return {
            "img1_path": str(Path(batch_name) / "dataset" / str(item["image_1"])),
            "img2_path": str(Path(batch_name) / "dataset" / str(item["image_2"])),
        }

    raise ValueError(f"Unsupported pair item keys: {sorted(item.keys())}")


class CleanNegativePairsDataset(Dataset):
    """
    Clean negative pairs dataset driven by JSON list.

    Output matches NegativeImagePairDataset:
      (img1, img2, img1_aug, img2_aug, seq1, seq2)
    """

    def __init__(
        self,
        negative_root: str,
        pairs_json_path: str,
        tokenizer: TransformTokenizer,
        transformer: ImageTransformer,
        split: str = "train",
        val_size: float = 0.1,
        random_seed: int = 42,
        max_seq_len: int = 15,
        augmentation_p: float = 0.5,
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
        if split == "val":
            self.pairs = shuffled[:n_val]
        else:
            self.pairs = shuffled[n_val:]

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

        seq1_tensor = seq1_ids.detach().clone().long()
        seq2_tensor = seq2_ids.detach().clone().long()

        return img1_t, img2_t, img1_aug_t, img2_aug_t, seq1_tensor, seq2_tensor


# def build_model_and_load_weights(cfg) -> ImageTransformPredictorViTQFormer:
#     model_cfg = cfg.model
#     model = ImageTransformPredictorViTQFormer(model_cfg)

#     ckpt_path = model_cfg.get("initialization_weights_path", None)
#     if ckpt_path is not None and os.path.isfile(ckpt_path):
#         state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
#         if isinstance(state, dict) and "model_state_dict" in state:
#             state = state["model_state_dict"]
#         incompatible = model.load_state_dict(state, strict=False)
#         print("Loaded base weights with IncompatibleKeys:", incompatible)
#     else:
#         print("No initialization weights found; tuning from random init.")

#     # Tune stage on clean set: размораживаем ВСЮ модель
#     for _, p in model.named_parameters():
#         p.requires_grad = True
#     print("Tune setup: all model parameters are trainable.")

#     return model

def build_model_and_load_weights(cfg, weights_path: str | None = None) -> ImageTransformPredictorViTQFormer:
    model_cfg = cfg.model
    model = ImageTransformPredictorViTQFormer(model_cfg)

    ckpt_path = weights_path or model_cfg.get("initialization_weights_path", None)
    if ckpt_path is not None and os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # поддержка как "голого" state_dict, так и чекпоинта с полем model_state_dict
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        incompatible = model.load_state_dict(state, strict=False)
        print("Loaded base weights with IncompatibleKeys:", incompatible)
    else:
        print("No base weights provided/found, training from scratch for new model.")

    # Pretrain на DomainNet: обучаем только Q-Former (query_compressor),
    # все остальные параметры замораживаем.
    for _, param in model.named_parameters():
        param.requires_grad = False
    for _, param in model.image_pair_encoder.query_compressor.named_parameters():
        param.requires_grad = True
    print("Pretrain setup: only image_pair_encoder.query_compressor is trainable.")

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Tune ImageTransformPredictorViTQFormer on cleaned negative pairs JSON."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config yaml (e.g. configs/train_config_vit_qformer_pretrain.yaml or tune config).",
    )
    parser.add_argument(
        "--negative-root",
        type=str,
        required=True,
        help="Root directory that contains batch folders (e.g. .../datasets).",
    )
    parser.add_argument(
        "--clean-json",
        type=str,
        required=True,
        help="Path to JSON with cleaned pairs list (must contain batch_name/image_1/image_2 OR dataset/img1/img2).",
    )
    parser.add_argument(
        "--augmentation-p",
        type=float,
        default=0.3,
        help="Augmentation probability used inside CleanNegativePairsDataset.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Validation split fraction.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split.",
    )

    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    tokenizer = TransformTokenizer()
    transformer = ImageTransformer()

    train_ds = CleanNegativePairsDataset(
        negative_root=args.negative_root,
        pairs_json_path=args.clean_json,
        tokenizer=tokenizer,
        transformer=transformer,
        split="train",
        val_size=args.val_size,
        random_seed=args.seed,
        max_seq_len=cfg.model.decoder.max_seq_len,
        augmentation_p=args.augmentation_p,
    )

    val_ds = CleanNegativePairsDataset(
        negative_root=args.negative_root,
        pairs_json_path=args.clean_json,
        tokenizer=tokenizer,
        transformer=transformer,
        split="val",
        val_size=args.val_size,
        random_seed=args.seed,
        max_seq_len=cfg.model.decoder.max_seq_len,
        augmentation_p=args.augmentation_p,
    )


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

    model = build_model_and_load_weights(cfg)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        tokenizer=tokenizer,
    )


if __name__ == "__main__":
    main()