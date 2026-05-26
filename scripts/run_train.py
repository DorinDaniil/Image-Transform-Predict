#!/usr/bin/env python3
import argparse
from src.dataset import ImageTransformer, AugmentationScheduler
from src.dataset import TransformTokenizer
from src.dataset import get_domainnet_dataloaders
from src.model import ImageTransformPredictor
from src.train_complex import train_model
from omegaconf import OmegaConf


def main(data_path):
    config_path = "configs/train_config_011_complex.yaml"
    train_config = OmegaConf.load(config_path)

    # Initialize model
    model = ImageTransformPredictor(train_config.model)
    init_weights = train_config.model.get('initialization_weights_path')
    if init_weights is not None:
        print(f"Loading pretrained weights from: {init_weights}")
        state_dict = torch.load(init_weights, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)

    tokenizer = TransformTokenizer()
    transformer = ImageTransformer()

    aug_config = train_config.get('augmentation', {})
    augmentation_scheduler = AugmentationScheduler(
        initial_p=aug_config.get('initial_p', 0.3),
        milestones=aug_config.get('milestones', []),
        probs=aug_config.get('probs', [])
    )

    dataloaders = get_domainnet_dataloaders(
        data_path,
        tokenizer,
        transformer,
        batch_size=train_config.training.batch_size,
        augmentation_scheduler=augmentation_scheduler
    )

    train_model(
        model,
        dataloaders['train'],
        dataloaders['val'],
        train_config,
        augmentation_scheduler
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Image Transform Predictor.')
    parser.add_argument(
        '--data_path',
        type=str,
        default="/mnt/DATA2/dorin/Image-Transform-Predict/data",
        help='Full path to the dataset directory'
    )
    args = parser.parse_args()
    main(args.data_path)