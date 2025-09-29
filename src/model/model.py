import torch
import torch.nn as nn
from omegaconf import DictConfig
from typing import Optional, Tuple
from .encoder import ImagePairEncoder
from .decoder import TransformDecoder
from .tokenizer import START_TOKEN_ID, END_TOKEN_ID, PAD_TOKEN_ID


class ImageTransformPredictor(nn.Module):
    """
    Complete end-to-end model for predicting image transformation sequences from image pairs.

    Architecture:
        1. ImagePairEncoder: Extracts fused features from two images using EfficientNet-B3
        2. TransformDecoder: Autoregressively generates sequence of transformation tokens

    Input:
        - image_batch_1, image_batch_2: [B, 3, 224, 224]
        - idx: [B, L] — full token sequence including START (e.g., [START, A, B, END, PAD, ...])

    Special Tokens:
        [PAD] = 0, [START] = 1, [END] = 2

    Usage:
        config = OmegaConf.load("config.yaml")
        model = ImageTransformPredictor(config.model)
        logits, loss = model(img1, img2, idx=full_sequence)  # training
        generated = model.generate(img1, img2)               # inference
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.image_pair_encoder = ImagePairEncoder(config.encoder)
        self.transform_decoder = TransformDecoder(config.decoder)

    def forward(
        self,
        image_batch_1: torch.Tensor,
        image_batch_2: torch.Tensor,
        idx: torch.LongTensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass. Automatically constructs targets from idx for loss computation.

        Args:
            image_batch_1: [B, 3, 224, 224] — source image
            image_batch_2: [B, 3, 224, 224] — target image
            idx: [B, L] — full input sequence including START token.
                 Example: [START, A, B, END, PAD, PAD]

        Returns:
            logits: [B, L, vocab_size] — predictions for next token at each position
            loss: scalar tensor (or None if in eval mode and no targets needed)
        """
        # Encode image pair
        combined_embedding = self.image_pair_encoder(image_batch_1, image_batch_2)

        # Construct targets: shift idx left by 1, pad last position
        targets = torch.roll(idx, shifts=-1, dims=1)
        targets[:, -1] = PAD_TOKEN_ID  # no next token after last --> PAD

        # Forward through decoder
        logits, loss = self.transform_decoder(
            idx=idx,
            combined_embedding=combined_embedding,
            targets=targets
        )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        image_batch_1: torch.Tensor,
        image_batch_2: torch.Tensor,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.LongTensor:
        """Autoregressive generation."""
        if max_new_tokens is None:
            max_new_tokens = self.config.decoder.max_seq_len - 1

        combined_embedding = self.image_pair_encoder(image_batch_1, image_batch_2)
        return self.transform_decoder.generate(
            combined_embedding=combined_embedding,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )