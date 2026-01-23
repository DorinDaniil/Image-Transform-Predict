import os
import random
from typing import List, Optional, Tuple, Callable, Dict
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .tokenizer import TransformTokenizer
from .augmentation import ImageTransformer, AugmentationScheduler


def _default_image_preprocessor() -> Callable[[Image.Image], torch.Tensor]:
    """Default image preprocessor: resize to 224x224 + ImageNet normalization."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# def _default_image_preprocessor() -> Callable[[Image.Image], torch.Tensor]:
#     """Default image preprocessor: resize to 300x300 + ImageNet normalization."""
#     return transforms.Compose([
#         transforms.Resize((300, 300)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])


class DomainNetDataset(Dataset):
    """
    DomainNet dataset with per-domain train/val split.
    
    Returns:
        - original_image (tensor)
        - transformed_image (tensor)
        - tokenized augmentation sequence (tensor)
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer: TransformTokenizer,
        transformer: ImageTransformer,
        split: str = "train",
        val_size: float = 0.1,
        image_preprocessor: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        random_seed: int = 42,
        max_seq_len: int = 15,
        augmentation_scheduler: Optional[AugmentationScheduler] = None,
    ):
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'!")

        self.data_dir = data_dir
        self.preprocessor = image_preprocessor or _default_image_preprocessor()
        self.tokenizer = tokenizer
        self.transformer = transformer
        self.split = split
        self.max_seq_len = max_seq_len
        self.augmentation_scheduler = augmentation_scheduler

        domain_to_paths: Dict[str, List[str]] = {}
        for domain in sorted(os.listdir(data_dir)):
            domain_path = os.path.join(data_dir, domain)
            if not os.path.isdir(domain_path):
                continue
            paths = []
            for file in os.listdir(domain_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    paths.append(os.path.join(domain_path, file))
            if paths:
                domain_to_paths[domain] = paths

        self.image_paths = []
        rng = random.Random(random_seed)

        for domain, paths in domain_to_paths.items():
            shuffled = rng.sample(paths, len(paths))
            n_val = int(len(shuffled) * val_size)

            if split == "val":
                self.image_paths.extend(shuffled[:n_val])
            else:  # train
                self.image_paths.extend(shuffled[n_val:])

        if split == "train":
            self.image_paths = rng.sample(self.image_paths, len(self.image_paths))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[idx]
        original_image = Image.open(image_path).convert("RGB")

        p = self.augmentation_scheduler.p if self.augmentation_scheduler is not None else None

        transformed_image, transform_sequence = self.transformer.transform(original_image, p=p)

        original_tensor = self.preprocessor(original_image)
        transformed_tensor = self.preprocessor(transformed_image.convert("RGB"))

        sequence_ids = self.tokenizer.encode(
            transforms=transform_sequence,
            add_special_tokens=True,
            max_seq_len=self.max_seq_len,
            return_targets=False,
        )

        return original_tensor, transformed_tensor, sequence_ids


def get_domainnet_dataloaders(
    data_dir: str,
    tokenizer: TransformTokenizer,
    transformer: ImageTransformer,
    batch_size: int = 32,
    num_workers: int = 4,
    val_size: float = 0.1,
    image_preprocessor: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    random_seed: int = 42,
    max_seq_len: int = 15,
    augmentation_scheduler: Optional[AugmentationScheduler] = None,
):
    common_kwargs = dict(
        data_dir=data_dir,
        tokenizer=tokenizer,
        transformer=transformer,
        val_size=val_size,
        image_preprocessor=image_preprocessor,
        random_seed=random_seed,
        max_seq_len=max_seq_len,
        augmentation_scheduler=augmentation_scheduler,
    )

    train_dataset = DomainNetDataset(split="train", **common_kwargs)
    val_dataset = DomainNetDataset(split="val", **common_kwargs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {"train": train_loader, "val": val_loader}


class SimpleDomainNetDataset(Dataset):
    """
    A simplified DomainNet dataset that returns:
    - PIL images (without preprocessing),
    - class labels for each image,
    - and stores the domain distribution.

    Args:
        data_dir (str): Path to the directory containing DomainNet images.
        split (str): Dataset split, either "train" or "val". Default: "train".
        val_size (float): Fraction of data to use for validation. Default: 0.1.
        random_seed (int): Random seed for reproducibility. Default: 42.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        val_size: float = 0.1,
        random_seed: int = 42,
    ):
        if split not in ("train", "val"):
            raise ValueError("Split must be 'train' or 'val'!")

        self.data_dir = data_dir
        self.split = split
        self.domain_distribution: Dict[str, int] = {}  # Stores domain-wise image counts

        # Collect image paths and labels
        self.image_paths: List[Tuple[str, int]] = []
        self._collect_images_and_labels(val_size, random_seed)

        # Calculate domain distribution
        self._calculate_domain_distribution()

    def _collect_images_and_labels(self, val_size: float, random_seed: int) -> None:
        """
        Collects image paths and their corresponding labels.

        Args:
            val_size (float): Fraction of data to use for validation.
            random_seed (int): Random seed for reproducibility.
        """
        domain_to_paths: Dict[str, List[Tuple[str, int]]] = {}
        rng = random.Random(random_seed)

        # Iterate over each domain directory
        for domain in sorted(os.listdir(self.data_dir)):
            domain_path = os.path.join(self.data_dir, domain)
            if not os.path.isdir(domain_path):
                continue

            # Collect all image paths and labels for the domain
            paths_with_labels = []
            for file in os.listdir(domain_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Extract label from filename or metadata (replace with actual logic)
                    # Example: If label is part of the filename or loaded from JSON
                    label = 0  # Replace with actual label extraction logic
                    paths_with_labels.append((os.path.join(domain_path, file), label))

            if paths_with_labels:
                domain_to_paths[domain] = paths_with_labels

        # Split into train/val
        for domain, paths in domain_to_paths.items():
            shuffled = rng.sample(paths, len(paths))
            n_val = int(len(shuffled) * val_size)

            if self.split == "val":
                self.image_paths.extend(shuffled[:n_val])
            else:  # train
                self.image_paths.extend(shuffled[n_val:])

        # Shuffle train split
        if self.split == "train":
            self.image_paths = rng.sample(self.image_paths, len(self.image_paths))

    def _calculate_domain_distribution(self) -> None:
        """Calculates and stores the domain-wise distribution of images."""
        for path, _ in self.image_paths:
            domain = os.path.basename(os.path.dirname(path))
            self.domain_distribution[domain] = self.domain_distribution.get(domain, 0) + 1

    def __len__(self) -> int:
        """Returns the number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        """
        Returns a PIL image and its label.

        Args:
            idx (int): Index of the image.

        Returns:
            Tuple[Image.Image, int]: PIL image and its class label.
        """
        image_path, label = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        return image, label
