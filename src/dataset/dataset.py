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