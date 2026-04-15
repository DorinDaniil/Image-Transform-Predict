"""
Loss helpers for the plagiarism-aware pair model.

Loss composition used by training/tuning scripts:
    total = w_seq * seq_ce + w_bce * bce + w_nce * info_nce

Where:
    - seq_ce: autoregressive CE over the transform-sequence decoder.
    - bce: binary cross-entropy on the match-head logit ({0, 1} labels).
    - info_nce: symmetric NT-Xent on projected per-image global features.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    Small 2-layer MLP projection head used for InfoNCE/NT-Xent.
    Projects raw per-image features (e.g. 1536-dim EffNet features) into a
    smaller L2-normalised space suitable for contrastive learning.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_dim] -> [B, out_dim] (L2-normalised)."""
        z = self.net(x)
        return F.normalize(z, dim=-1)


def info_nce_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Symmetric NT-Xent / InfoNCE.

    Args:
        z_a: [B, D] L2-normalised anchor features.
        z_b: [B, D] L2-normalised positive features (positive for row i is row i in z_b).
        temperature: scaling.

    Returns:
        Scalar loss.
    """
    if z_a.shape != z_b.shape:
        raise ValueError(f"z_a and z_b must match, got {z_a.shape} vs {z_b.shape}")

    sim = z_a @ z_b.T / temperature  # [B, B]
    labels = torch.arange(z_a.size(0), device=z_a.device)
    loss_ab = F.cross_entropy(sim, labels)
    loss_ba = F.cross_entropy(sim.T, labels)
    return 0.5 * (loss_ab + loss_ba)


def pairwise_bce_loss(
    match_logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Binary cross-entropy on match logits.

    Args:
        match_logits: [B] raw logits (pre-sigmoid).
        labels: [B] {0, 1}.
        pos_weight: optional tensor scalar used by BCEWithLogits to rebalance
            positive class (useful when negatives dominate).
    """
    return F.binary_cross_entropy_with_logits(
        match_logits, labels.float(), pos_weight=pos_weight
    )


class SeqTokenAccuracy:
    """Epoch-level macro-averaged per-token accuracy (PAD tokens excluded)."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id
        self.reset()

    def reset(self) -> None:
        self.total_correct = 0
        self.total_tokens = 0

    def update(
        self, seq_logits: torch.Tensor, idx: torch.Tensor,
    ) -> None:
        """
        Args:
            seq_logits: [N, T, V] decoder output logits.
            idx: [N, T] teacher-forced input (BOS, tok1, tok2, ...).
        """
        targets = torch.roll(idx, shifts=-1, dims=1)
        targets[:, -1] = self.pad_token_id
        preds = seq_logits.detach().argmax(dim=-1)  # [N, T]
        mask = targets != self.pad_token_id
        self.total_correct += int((preds[mask] == targets[mask]).sum())
        self.total_tokens += int(mask.sum())

    @property
    def accuracy(self) -> float:
        return self.total_correct / self.total_tokens if self.total_tokens > 0 else 0.0


class SeqBinaryMetrics:
    """Binary classification derived from the decoder's first predicted token.

    Rule: if argmax(logits at position 0) == eos_token_id the decoder
    predicts "no transform" → class 0.  Otherwise → class 1.
    Accumulates TP/FP/FN/TN the same way as BinaryMatchMetrics.
    """

    def __init__(self, eos_token_id: int) -> None:
        self.eos_token_id = eos_token_id
        self.reset()

    def reset(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, seq_logits: torch.Tensor, labels: torch.Tensor) -> None:
        first_pred = seq_logits.detach()[:, 0, :].argmax(dim=-1)  # [N]
        preds = (first_pred != self.eos_token_id).long()
        labels = labels.detach().cpu().long()
        preds = preds.cpu()
        self.tp += int(((preds == 1) & (labels == 1)).sum())
        self.fp += int(((preds == 1) & (labels == 0)).sum())
        self.fn += int(((preds == 0) & (labels == 1)).sum())
        self.tn += int(((preds == 0) & (labels == 0)).sum())

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d > 0 else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r > 0 else 0.0

    @property
    def fpr(self) -> float:
        d = self.fp + self.tn
        return self.fp / d if d > 0 else 0.0


class BinaryMatchMetrics:
    """
    Epoch-level metrics using a CALIBRATED probability threshold on match_logit.
    Tracks precision/recall/F1 and FPR — the latter is the real optimisation
    target for plagiarism detection.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, match_logits: torch.Tensor, labels: torch.Tensor) -> None:
        probs = torch.sigmoid(match_logits.detach()).cpu()
        preds = (probs >= self.threshold).long()
        labels = labels.detach().cpu().long()
        self.tp += int(((preds == 1) & (labels == 1)).sum())
        self.fp += int(((preds == 1) & (labels == 0)).sum())
        self.fn += int(((preds == 0) & (labels == 1)).sum())
        self.tn += int(((preds == 0) & (labels == 0)).sum())

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r > 0 else 0.0

    @property
    def fpr(self) -> float:
        denom = self.fp + self.tn
        return self.fp / denom if denom > 0 else 0.0
