"""
End-to-end plagiarism-aware pair model.

Composition:
    ImagePairEncoderPlagiarism  -- EffNet-B3 (global features only) +
                                   interaction-aware multi-token fuser [B, K, D]
    TransformDecoder            -- existing autoregressive decoder reused as-is
    BinaryMatchHead             -- explicit binary classification head
    ProjectionHead              -- small MLP for InfoNCE over raw features
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig

from ..decoder import TransformDecoder
from .encoder import ImagePairEncoderPlagiarism
from .head import BinaryMatchHead
from .losses import ProjectionHead


class ImageTransformPlagiarismPredictor(nn.Module):
    """
    Plagiarism-oriented pair predictor.

    Forward outputs (in this order):
        seq_logits:   [B, T, vocab_size]
        seq_loss:     scalar (CE over non-pad target tokens)
        match_logit:  [B] raw binary logit per pair
        fused_tokens: [B, K, D]  (fused context, detached-friendly for downstream)

    extract_image_embeddings returns per-image features BEFORE fusion, enabling
    BxB cross-pairing in pretraining without wasted backbone passes.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config

        self.image_pair_encoder = ImagePairEncoderPlagiarism(config.encoder)

        self.bos_token_id = int(config.decoder.bos_token_id)
        self.eos_token_id = int(config.decoder.eos_token_id)
        self.pad_token_id = int(config.decoder.pad_token_id)

        self.transform_decoder = TransformDecoder(config.decoder)

        match_cfg = getattr(config, "match_head", None)
        match_hidden = (
            int(match_cfg.hidden_dim)
            if match_cfg is not None and hasattr(match_cfg, "hidden_dim")
            else int(config.decoder.n_embd)
        )
        match_dropout = (
            float(match_cfg.dropout)
            if match_cfg is not None and hasattr(match_cfg, "dropout")
            else 0.1
        )
        self.match_head = BinaryMatchHead(
            in_dim=int(config.decoder.n_embd),
            hidden_dim=match_hidden,
            dropout=match_dropout,
        )

        # Projection head for InfoNCE on RAW per-image features.
        # Kept optional — only used when training scripts enable it.
        proj_cfg = getattr(config, "projection_head", None)
        proj_hidden = (
            int(proj_cfg.hidden_dim)
            if proj_cfg is not None and hasattr(proj_cfg, "hidden_dim")
            else 512
        )
        proj_out = (
            int(proj_cfg.out_dim)
            if proj_cfg is not None and hasattr(proj_cfg, "out_dim")
            else 128
        )
        self.projection_head = ProjectionHead(
            in_dim=self.image_pair_encoder.feature_dim,
            hidden_dim=proj_hidden,
            out_dim=proj_out,
        )

    def extract_image_embeddings(
        self, image_batch_1: torch.Tensor, image_batch_2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.image_pair_encoder.extract_image_embeddings(
            image_batch_1, image_batch_2
        )

    def project_features(self, features: torch.Tensor) -> torch.Tensor:
        """Project [B, 1, D_feat] or [B, D_feat] features to L2-normalised [B, D_proj]."""
        if features.dim() == 3:
            features = features.squeeze(1)
        return self.projection_head(features)

    def forward(
        self,
        image_batch_1: torch.Tensor,
        image_batch_2: torch.Tensor,
        idx: torch.LongTensor,
        use_precomputed_embeddings: bool = False,
        return_match_logit: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        fused = self.image_pair_encoder(
            image_batch_1, image_batch_2, use_precomputed_embeddings
        )  # [B, K, D]

        targets = torch.roll(idx, shifts=-1, dims=1)
        targets[:, -1] = self.pad_token_id
        seq_logits, seq_loss = self.transform_decoder(
            idx=idx,
            images_embeddings=fused,
            targets=targets,
        )

        match_logit = self.match_head(fused) if return_match_logit else None
        return seq_logits, seq_loss, match_logit, fused

    @torch.no_grad()
    def generate(
        self,
        image_batch_1: torch.Tensor,
        image_batch_2: torch.Tensor,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        do_sample: bool = False,
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        """
        Autoregressive generation + binary match probability.

        Returns:
            token_ids: [B, 1 + max_new_tokens]
            match_prob: [B] sigmoid(match_head) probability
        """
        if max_new_tokens is None:
            max_new_tokens = self.config.decoder.max_seq_len - 1

        pad_token_id = pad_token_id if pad_token_id is not None else self.pad_token_id
        bos_token_id = bos_token_id if bos_token_id is not None else self.bos_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.eos_token_id

        fused = self.image_pair_encoder(image_batch_1, image_batch_2)  # [B, K, D]
        match_prob = torch.sigmoid(self.match_head(fused))

        token_ids = self.transform_decoder.generate(
            images_embeddings=fused,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )
        return token_ids, match_prob
