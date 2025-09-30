#!/usr/bin/env python3
import argparse
from src.dataset import get_domainnet_dataloaders
from src.dataset import ImageTransformer, TransformTokenizer
from src.model import ImageTransformPredictor
from src.train import train_model
from omegaconf import OmegaConf

def main(data_path):
    config_path = "configs/train_config.yaml"
    train_config = OmegaConf.load(config_path)

    model = ImageTransformPredictor(train_config.model)
    tokenizer = TransformTokenizer()
    transformer = ImageTransformer()
    dataloaders = get_domainnet_dataloaders(data_path, tokenizer, transformer, batch_size=train_config.training.batch_size)

    train_model(model, dataloaders['train'], dataloaders['val'], train_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Image Transform Predictor.')
    parser.add_argument('--data_path', type=str, default="/home/jovyan/nkiselev/ddorin/project/Image-Transform-Predict/src/data", help='Full path to the dataset directory')
    args = parser.parse_args()

    main(args.data_path)