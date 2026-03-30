import os
import random
from pathlib import Path
from typing import List, Optional, Callable, Tuple, Dict, Union
from PIL import Image
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .tokenizer import TransformTokenizer
from .augmentation import ImageTransformer


# ==============================================================================
# Preprocessors
# ==============================================================================
def _default_image_preprocessor() -> Callable[[Image.Image], torch.Tensor]:
    """Default image preprocessor: resize to 224x224 + ImageNet normalization."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# def _default_image_preprocessor() -> Callable[[Image.Image], torch.Tensor]:
#     """Alternative: resize to 300x300 + ImageNet normalization."""
#     return transforms.Compose([
#         transforms.Resize((300, 300)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])


# ==============================================================================
# PyTorch Dataset for negative image pairs with augmentation & sequences
# ==============================================================================

class NegativeImagePairDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        batch_names: List[str],
        tokenizer: TransformTokenizer,
        transformer: ImageTransformer,
        split: str = "train",
        val_size: float = 0.1,
        image_preprocessor: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        random_seed: int = 42,
        max_seq_len: int = 15,
        augmentation_p: float = 0.5,
        return_metadata: bool = False,
    ):
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'!")
        
        self.root_dir = Path(root_dir)
        self.tokenizer = tokenizer
        self.transformer = transformer
        self.split = split
        self.max_seq_len = max_seq_len
        self.augmentation_p = augmentation_p
        self.return_metadata = return_metadata
        self.preprocessor = image_preprocessor or _default_image_preprocessor()

        # --- Load all candidate pairs first ---
        all_pairs: List[Tuple[Path, Path, Optional[Dict]]] = []  # (img1, img2, meta)
        
        for batch_name in batch_names:
            batch_path = self.root_dir / batch_name
            img_dir = batch_path / "dataset"
            meta_path = batch_path / "metadata.json"

            if not img_dir.is_dir():
                raise ValueError(f"Image dir not found: {img_dir}")
            
            if self.return_metadata:
                if not meta_path.is_file():
                    raise FileNotFoundError(f"metadata.json required but missing in {batch_path}")
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        raw_metadata = json.load(f)
                    if isinstance(raw_metadata, dict):
                        raw_metadata = list(raw_metadata.values())
                except Exception as e:
                    raise ValueError(f"Failed to load {meta_path}: {e}")
                
                for item in raw_metadata:
                    img1_name = item.get("image_1")
                    img2_name = item.get("image_2")
                    if not img1_name or not img2_name:
                        continue
                    img1_path = img_dir / img1_name
                    img2_path = img_dir / img2_name
                    if img1_path.is_file() and img2_path.is_file():
                        all_pairs.append((img1_path, img2_path, {
                            "image_1": img1_name,
                            "image_2": img2_name,
                            "batch_name": batch_name,
                        }))
            else:
                pairs = list_image_pairs(img_dir)
                for img1_path, img2_path in pairs:
                    all_pairs.append((img1_path, img2_path, None))

        if not all_pairs:
            raise ValueError("No valid image pairs loaded.")

        # --- Train/val split ---
        rng = random.Random(random_seed)
        shuffled = rng.sample(all_pairs, len(all_pairs))
        n_val = int(len(shuffled) * val_size)
        if split == "val":
            self._pairs_and_meta = shuffled[:n_val]
        else:  # train
            self._pairs_and_meta = shuffled[n_val:]
            # optional: further shuffle train
            self._pairs_and_meta = rng.sample(self._pairs_and_meta, len(self._pairs_and_meta))

    def __len__(self) -> int:
        return len(self._pairs_and_meta)

    def __getitem__(self, idx: int) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]
    ]:
        img1_path, img2_path, meta = self._pairs_and_meta[idx]

        # Load originals
        img1_orig = Image.open(img1_path).convert("RGB")
        img2_orig = Image.open(img2_path).convert("RGB")

        # Augment both with *independent* transforms (important for negatives)
        img1_aug, seq1 = self.transformer.transform(img1_orig, p=self.augmentation_p)
        img2_aug, seq2 = self.transformer.transform(img2_orig, p=self.augmentation_p)

        # Preprocess all four
        img1_t = self.preprocessor(img1_orig)
        img2_t = self.preprocessor(img2_orig)
        img1_aug_t = self.preprocessor(img1_aug.convert("RGB"))
        img2_aug_t = self.preprocessor(img2_aug.convert("RGB"))

        # Tokenize sequences — assume tokenizer returns torch.Tensor
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

        if self.return_metadata:
            return img1_t, img2_t, img1_aug_t, img2_aug_t, seq1_tensor, seq2_tensor, meta
        else:
            return img1_t, img2_t, img1_aug_t, img2_aug_t, seq1_tensor, seq2_tensor


# ==============================================================================
# Helper to list image pairs if no metadata used
# ==============================================================================

def list_image_pairs(dataset_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Returns a list of image path pairs (img1, img2) from the given directory.
    Expects filenames in the format: 000000001_1.png, 000000001_2.png, etc.

    Args:
        dataset_dir (Path): Path to the directory containing image pairs.

    Returns:
        List[Tuple[Path, Path]]: List of tuples, where each tuple contains paths to two images forming a pair.

    Raises:
        ValueError: If the provided path is not a valid directory.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise ValueError(f"Directory does not exist: {dataset_dir}")

    first_files = sorted(dataset_dir.glob("*_1.png"))
    pairs = []
    for f1 in first_files:
        stem = f1.stem  # e.g., "000000001_1"
        idx = stem.rsplit('_', 1)[0]  # e.g., "000000001"
        f2 = dataset_dir / f"{idx}_2.png"
        if f2.exists():
            pairs.append((f1, f2))
        else:
            print(f"Warning: missing pair for {f1}")
    return pairs


# ==============================================================================
# Function for getting negative pair dataloaders
# ==============================================================================

def get_negative_pair_dataloaders(
    root_dir: str,
    batch_names: List[str],
    tokenizer: TransformTokenizer,
    transformer: ImageTransformer,
    batch_size: int = 32,
    num_workers: int = 4,
    val_size: float = 0.1,
    image_preprocessor: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    random_seed: int = 42,
    max_seq_len: int = 15,
    augmentation_p: float = 0.5,
    return_metadata: bool = False,
    pin_memory: bool = True,
    drop_last_train: bool = True,
):
    """
    Creates train and validation DataLoaders for negative image pair dataset.

    Returns:
        Dict[str, DataLoader]: {"train": train_loader, "val": val_loader}
    """
    common_kwargs = dict(
        root_dir=root_dir,
        batch_names=batch_names,
        tokenizer=tokenizer,
        transformer=transformer,
        val_size=val_size,
        image_preprocessor=image_preprocessor,
        random_seed=random_seed,
        max_seq_len=max_seq_len,
        augmentation_p=augmentation_p,
        return_metadata=return_metadata,
    )

    train_dataset = NegativeImagePairDataset(split="train", **common_kwargs)
    val_dataset = NegativeImagePairDataset(split="val", **common_kwargs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_train,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return {"train": train_loader, "val": val_loader}
