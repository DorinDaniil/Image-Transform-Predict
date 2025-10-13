import torch
import torch.nn as nn
from omegaconf import DictConfig
from typing import Optional, Tuple
from .efficientnet_encoder import ImagePairEfficientNet
from .decoder import TransformDecoder
from .vit_encoder import ImagePairViT

class ImageTransformPredictor(nn.Module):
    """
    Complete end-to-end model for predicting image transformation sequences from image pairs.
    """
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        encoder_type = config.encoder.get("type", "efficientnet")
        if encoder_type == "vit":
            self.image_pair_encoder = ImagePairViT(config.encoder)
        else:
            self.image_pair_encoder = ImagePairEfficientNet(config.encoder)

        self.bos_token_id = config.decoder.bos_token_id
        self.eos_token_id = config.decoder.eos_token_id
        self.pad_token_id = config.decoder.pad_token_id
        
        self.transform_decoder = TransformDecoder(config.decoder)

    def extract_image_embeddings(self, image_batch_1, image_batch_2):
        """
        Extract embeddings for two batches of images.
        Args:
            image_batch_1 (torch.Tensor): First batch of images with shape [batch_size, 3, 224, 224].
            image_batch_2 (torch.Tensor): Second batch of images with shape [batch_size, 3, 224, 224].
        Returns:
            tuple: (features_1, features_2) — two batches of embeddings.
        """
        return self.image_pair_encoder.extract_image_embeddings(image_batch_1, image_batch_2)

    def forward(
        self,
        image_batch_1: torch.Tensor,
        image_batch_2: torch.Tensor,
        idx: torch.LongTensor,
        use_precomputed_embeddings: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        """
        combined_embedding = self.image_pair_encoder(image_batch_1, image_batch_2, use_precomputed_embeddings)

        targets = torch.roll(idx, shifts=-1, dims=1)
        targets[:, -1] = self.pad_token_id
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
        do_sample: bool = False,
        pad_token_id: int = None,
        bos_token_id: int = None,
        eos_token_id: int = None,
    ) -> torch.LongTensor:
        """Autoregressive generation."""
        if max_new_tokens is None:
            max_new_tokens = self.config.decoder.max_seq_len - 1

        if pad_token_id is None:
            pad_token_id = self.pad_token_id
        if bos_token_id is None:
            bos_token_id = self.bos_token_id
        if eos_token_id is None:
            eos_token_id = self.eos_token_id

        combined_embedding = self.image_pair_encoder(image_batch_1, image_batch_2)
        return self.transform_decoder.generate(
            combined_embedding=combined_embedding,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id
        )