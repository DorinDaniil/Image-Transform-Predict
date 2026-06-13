"""
Binary match head.

Operates on the set of fused pair-tokens ``[B, K, D]`` produced by
``MultiTokenPairFuser`` and emits one logit per pair indicating
probability of "positive match" (i.e. near-duplicate / plagiarism).
"""

from typing import Optional

import torch
import torch.nn as nn


class BinaryMatchHead(nn.Module):
    """
    Pools K fused tokens with both mean and max, then predicts a single logit.
    The mean+max combo improves separation when one of the K tokens carries
    most of the interaction signal.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else in_dim
        self.net = nn.Sequential(
            nn.Linear(2 * in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, K, D]

        Returns:
            match_logit: [B] raw logits (pre-sigmoid).
        """
        if tokens.dim() != 3:
            raise ValueError(f"tokens must be [B, K, D], got {tokens.shape}")
        mean_pool = tokens.mean(dim=1)      # [B, D]
        max_pool, _ = tokens.max(dim=1)     # [B, D]
        pooled = torch.cat([mean_pool, max_pool], dim=-1)  # [B, 2D]
        return self.net(pooled).squeeze(-1)
