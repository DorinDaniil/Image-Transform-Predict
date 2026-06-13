"""
Encoder for plagiarism-aware pair comparison.

Design constraints (from user):
  - Only GLOBAL pooled features of EfficientNet are used (no feature maps).
  - Output is a SMALL set of fused tokens (K tokens, K small) to be consumed
    by a transformer decoder via cross-attention.
  - Pair interaction is made explicit via [f1, f2, |f1 - f2|, f1 * f2].
"""

from typing import Tuple

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from omegaconf import DictConfig


class EfficientNetGlobalEncoder(nn.Module):
    """
    Global-pooled EfficientNet-B3 feature extractor.

    Returns a single 1536-dim vector per image, shaped as a length-1 sequence
    ``[B, 1, 1536]`` to keep API-compatibility with the rest of the codebase.
    No spatial feature maps are exposed by design.
    """

    FEATURE_DIM: int = 1536

    def __init__(self, freeze: bool = True) -> None:
        super().__init__()
        self.backbone = EfficientNet.from_pretrained("efficientnet-b3")
        self.backbone._dropout = nn.Identity()
        self.backbone._fc = nn.Identity()

        for param in self.backbone.parameters():
            param.requires_grad = not freeze

        self.feature_dim = self.FEATURE_DIM

    def set_trainable(self, trainable: bool) -> None:
        """Toggle backbone trainability (used for differential-LR warmup)."""
        for param in self.backbone.parameters():
            param.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)  # [B, 1536]
        return features.unsqueeze(1)  # [B, 1, 1536]


class MultiTokenPairFuser(nn.Module):
    """
    Fuses two global image features into K interaction-aware tokens.

    Interaction features:
        concat([f1, f2, |f1 - f2|, f1 * f2])  -> 4 * feature_dim
    Projected and reshaped into K tokens of dim out_dim.
    Each token gets a learned role-embedding (token-type).
    """

    def __init__(
        self,
        feature_dim: int = 1536,
        n_tokens: int = 4,
        out_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_tokens < 1:
            raise ValueError("n_tokens must be >= 1")

        self.feature_dim = feature_dim
        self.n_tokens = n_tokens
        self.out_dim = out_dim

        interaction_dim = 4 * feature_dim  # [f1, f2, |f1-f2|, f1*f2]

        self.proj = nn.Sequential(
            nn.Linear(interaction_dim, interaction_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(interaction_dim // 2, n_tokens * out_dim),
        )

        # Learnable role-embedding per token (acts as "token-type")
        self.role = nn.Parameter(torch.zeros(n_tokens, out_dim))
        nn.init.normal_(self.role, mean=0.0, std=0.02)

        self.norm = nn.LayerNorm(out_dim)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f1: [B, feature_dim]
            f2: [B, feature_dim]

        Returns:
            tokens: [B, n_tokens, out_dim]
        """
        if f1.dim() != 2 or f2.dim() != 2:
            raise ValueError(f"f1/f2 must be [B, D], got {f1.shape} and {f2.shape}")

        diff = torch.abs(f1 - f2)
        prod = f1 * f2
        interaction = torch.cat([f1, f2, diff, prod], dim=-1)  # [B, 4*D]

        fused = self.proj(interaction)  # [B, K*out_dim]
        tokens = fused.view(-1, self.n_tokens, self.out_dim)  # [B, K, out_dim]
        tokens = tokens + self.role.unsqueeze(0)
        tokens = self.norm(tokens)
        return tokens


class ImagePairEncoderPlagiarism(nn.Module):
    """
    End-to-end pair encoder: EfficientNet global features -> multi-token fuser.

    Expected config fields:
        freeze_image_encoder (bool): freeze backbone.
        out_token_n_embd (int): fused token dim (matches decoder n_embd).
        n_fused_tokens (int, optional, default=4): number of fused tokens K.
        fuser_dropout (float, optional, default=0.1).

    The ``forward`` method supports ``use_precomputed_embeddings``: in that case
    the arguments are expected to be per-image features ``[B, 1, feature_dim]``
    already extracted by ``extract_image_embeddings``. This mirrors the existing
    ``ImagePairEncoderEfficientNet`` API and is needed for BxB cross-pairing in
    pretraining.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config

        freeze = bool(config.freeze_image_encoder)
        self.image_encoder = EfficientNetGlobalEncoder(freeze=freeze)

        if not hasattr(config, "out_token_n_embd"):
            raise ValueError("encoder config must contain 'out_token_n_embd'")

        n_tokens = int(getattr(config, "n_fused_tokens", 4))
        dropout = float(getattr(config, "fuser_dropout", 0.1))

        self.fuser = MultiTokenPairFuser(
            feature_dim=self.image_encoder.feature_dim,
            n_tokens=n_tokens,
            out_dim=int(config.out_token_n_embd),
            dropout=dropout,
        )

        self.n_tokens = n_tokens
        self.output_dim = int(config.out_token_n_embd)
        self.feature_dim = self.image_encoder.feature_dim

    def extract_image_embeddings(
        self, image_batch_1: torch.Tensor, image_batch_2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return per-image global features ``[B, 1, feature_dim]`` each."""
        features_1 = self.image_encoder(image_batch_1)
        features_2 = self.image_encoder(image_batch_2)
        return features_1, features_2

    def forward(
        self,
        image_batch_1: torch.Tensor,
        image_batch_2: torch.Tensor,
        use_precomputed_embeddings: bool = False,
    ) -> torch.Tensor:
        """
        Returns fused multi-token context: ``[B, K, out_dim]``.
        """
        if not use_precomputed_embeddings:
            f1, f2 = self.extract_image_embeddings(image_batch_1, image_batch_2)
        else:
            f1, f2 = image_batch_1, image_batch_2  # expected [B, 1, feature_dim]

        f1_vec = f1.squeeze(1)
        f2_vec = f2.squeeze(1)
        return self.fuser(f1_vec, f2_vec)
