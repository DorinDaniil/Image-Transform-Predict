import argparse
import os
from omegaconf import OmegaConf
import torch

from src.dataset import (
    TransformTokenizer,
    ImageTransformer,
    get_negative_pair_dataloaders
)
from src.model import SiamNet
from src.tuning_siamnet import train_model


def main(data_path, config_path, resume):
    # Load configuration
    train_config = OmegaConf.load(config_path)
    
    # Initialize model
    model = SiamNet()
    
    # Load pretrained weights if specified
    init_weights = train_config.model.get('initialization_weights_path')

    if init_weights is not None:
        print(f"Loading pretrained weights from: {init_weights}")
        state_dict = torch.load(init_weights, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
    
    # Initialize tokenizer & transformer (required by dataloader but not used by SiamNet directly)
    tokenizer = TransformTokenizer()
    transformer = ImageTransformer()
    
    # Define dataset batch names (same as in your generation pipeline)
    batch_names = [
        "dataset_0", "dataset_1", "dataset_5", "dataset_9",
        "dataset_2", "dataset_18", "dataset_27", "dataset_36",
        "dataset_45", "dataset_270", "dataset_281", "dataset_150",
        "dataset_180"
    ]
    
    # Get dataloaders for negative pair dataset
    dataloaders = get_negative_pair_dataloaders(
        root_dir=data_path,
        batch_names=batch_names,
        tokenizer=tokenizer,
        transformer=transformer,
        batch_size=train_config.training.batch_size,
        val_size=0.1,
        image_preprocessor=None,
        random_seed=42,
        max_seq_len=16,
        augmentation_p=0.3,
        return_metadata=False
    )
    
    # Start fine-tuning
    train_model(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        config=train_config,
        resume=resume
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune SiamNet on negative/positive image pairs.')
    
    parser.add_argument(
        '--data_path',
        type=str,
        default="/mnt/DATA2/dorin/res.cv.science.dataset.generation/datasets",
        help='Root directory containing batch_X folders with image pairs'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default="configs/train_config_siamnet.yaml",
        help='Path to fine-tuning configuration YAML file'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from latest checkpoint in checkpoint_dir'
    )
    
    args = parser.parse_args()
    
    main(args.data_path, args.config, args.resume)