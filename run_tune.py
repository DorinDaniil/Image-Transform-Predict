#!/usr/bin/env python3
import argparse
import torch
from omegaconf import OmegaConf

from src.dataset import (
    TransformTokenizer,
    ImageTransformer,
    get_negative_pair_dataloaders
)
from src.model import ImageTransformPredictor
from src.tuning import train_model


def main(data_path):
    config_path = "configs/train_config_011.yaml"
    train_config = OmegaConf.load(config_path)

    # Initialize model
    model = ImageTransformPredictor(train_config.model)
    init_weights = train_config.model.get('initialization_weights_path')
    if init_weights is not None:
        print(f"Loading pretrained weights from: {init_weights}")
        state_dict = torch.load(init_weights, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)

    # Initialize tokenizer & transformer
    tokenizer = TransformTokenizer()
    transformer = ImageTransformer()

    # Define dataset batch names
    batch_names = [
        "dataset_0", "dataset_1", "dataset_5", "dataset_9"
    ]

    # Get dataloaders for negative pair dataset
    dataloaders = get_negative_pair_dataloaders(
        root_dir=data_path,
        batch_names=batch_names,
        tokenizer=tokenizer,
        transformer=transformer,
        batch_size=train_config.training.batch_size,
        val_size=train_config.training.get('val_size', 0.1),
        image_preprocessor=None,
        random_seed=train_config.training.get('random_seed', 42),
        max_seq_len=train_config.model.decoder.max_seq_len,
        augmentation_p=train_config.augmentation.get('initial_p', 0.5),
        return_metadata=False
    )

    train_model(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        config=train_config,
        tokenizer=tokenizer
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train on Negative Image Pairs.')
    parser.add_argument(
        '--data_path',
        type=str,
        default="/mnt/DATA2/dorin/res.cv.science.dataset.generation/datasets",
        help='Root directory containing batch_X folders'
    )
    args = parser.parse_args()
    main(args.data_path)