"""
Plagiarism-aware pair model based on EfficientNet global features.

Public API:
    EfficientNetGlobalEncoder   -- frozen/trainable EffNet-B3 (global pool only).
    MultiTokenPairFuser         -- interaction-aware [B, K, D] fuser.
    ImagePairEncoderPlagiarism  -- encoder + fuser.
    BinaryMatchHead             -- explicit binary classification head.
    ProjectionHead              -- small MLP for InfoNCE over raw features.
    BinaryMatchMetrics          -- precision/recall/F1/FPR tracker.
    ImageTransformPlagiarismPredictor -- end-to-end model.
    info_nce_loss, pairwise_bce_loss -- loss helpers.
"""

from .encoder import (
    EfficientNetGlobalEncoder,
    MultiTokenPairFuser,
    ImagePairEncoderPlagiarism,
)
from .head import BinaryMatchHead
from .losses import (
    ProjectionHead,
    info_nce_loss,
    pairwise_bce_loss,
    BinaryMatchMetrics,
)
from .model import ImageTransformPlagiarismPredictor

__all__ = [
    "EfficientNetGlobalEncoder",
    "MultiTokenPairFuser",
    "ImagePairEncoderPlagiarism",
    "BinaryMatchHead",
    "ProjectionHead",
    "info_nce_loss",
    "pairwise_bce_loss",
    "BinaryMatchMetrics",
    "ImageTransformPlagiarismPredictor",
]
