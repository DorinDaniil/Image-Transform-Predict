#!/usr/bin/env python3
import argparse
from src.dataset import ImageTransformer
from src.dataset import TransformTokenizer
from src.dataset import get_domainnet_dataloaders
from src.model import SiamNet
from src.train_siamnet import train_model
from omegaconf import OmegaConf


def main(data_path):
    config_path = "configs/train_config_siamnet.yaml"
    train_config = OmegaConf.load(config_path)

    model = SiamNet()
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
        augmentation_scheduler=augmentation_scheduler,
        return_seq=False
    )

    train_model(model, dataloaders['train'], dataloaders['val'], train_config, resume=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SiamNet.')
    parser.add_argument(
        '--data_path',
        type=str,
        default="/mnt/DATA2/dorin/Image-Transform-Predict/data",
        help='Full path to the dataset directory'
    )
    args = parser.parse_args()
    main(args.data_path)