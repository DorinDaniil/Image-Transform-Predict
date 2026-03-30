import argparse
import os

import torch
from omegaconf import OmegaConf

from src.model import ImageTransformPredictorViTQFormer
from src.dataset.tokenizer import TransformTokenizer
from src.dataset.augmentation import ImageTransformer, AugmentationScheduler
from src.dataset.dataset import get_domainnet_dataloaders
from src.train import train_model




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
        description="Train ImageTransformPredictorViTQFormer on DomainNet (pretrain stage)."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to train_config_vit.yaml (or compatible config).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/mnt/DATA2/dorin/Image-Transform-Predict/data",
        help="Path to DomainNet root directory.",
    )

    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # === Data / tokenizer / augmentations ===
    tokenizer = TransformTokenizer()
    transformer = ImageTransformer()

    aug_cfg = cfg.get("augmentation", None)
    if aug_cfg is not None:
        aug_scheduler = AugmentationScheduler(
            initial_p=aug_cfg.get("initial_p", 0.3),
            milestones=aug_cfg.get("milestones", []),
            probs=aug_cfg.get("probs", []),
        )
    else:
        aug_scheduler = AugmentationScheduler(initial_p=0.3, milestones=[], probs=[])

    loaders = get_domainnet_dataloaders(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        transformer=transformer,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.get("num_workers", 4),
        val_size=cfg.get("val_size", 0.1),
        max_seq_len=cfg.model.decoder.max_seq_len,
        augmentation_scheduler=aug_scheduler,
        return_seq=True,
    )

    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # === Model ===
    # веса берём из config.model.initialization_weights_path
    model = build_model_and_load_weights(cfg, weights_path=None)

    # === Train===
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        augmentation_scheduler=aug_scheduler,
    )


if __name__ == "__main__":
    main()