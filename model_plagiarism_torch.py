"""
Standalone inference model for ImageTransformPredictor.

Split into 4 ONNX-exportable components with clean tensor-only forward():
    1. ImageEncoder         - EfficientNet-B3  [B,3,300,300] -> [B,1536]
    2. ImageFuser           - pair interaction  [B,1536] x2   -> [B,K,512]
    3. BinaryMatchHead      - classification    [B,K,512]     -> [B]
    4. TransformationDecoder- autoregressive    [B,K,512]+[B,T] -> [B,T,V]
"""

import io
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from PIL import Image
from torchvision import transforms


preprocess = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# =============================================================================
# 1. ImageEncoder
# =============================================================================

class ImageEncoder(nn.Module):
    """
    EfficientNet-B3 global-pooled feature extractor.
    """

    FEATURE_DIM: int = 1536

    def __init__(self) -> None:
        super().__init__()
        self.backbone = EfficientNet.from_name("efficientnet-b3")
        self.backbone._dropout = nn.Identity()
        self.backbone._fc = nn.Identity()
        self.feature_dim = self.FEATURE_DIM

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 3, 300, 300] -> [B, 1536]."""
        return self.backbone(x)


# =============================================================================
# 2. ImageFuser
# =============================================================================

class ImageFuser(nn.Module):
    """
    Interaction-aware multi-token pair fuser.

    concat([f1, f2, |f1-f2|, f1*f2]) -> project -> [B, K, out_dim].
    """

    def __init__(
        self,
        feature_dim: int = 1536,
        n_tokens: int = 4,
        out_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.n_tokens = n_tokens
        self.out_dim = out_dim

        interaction_dim = 4 * feature_dim

        self.proj = nn.Sequential(
            nn.Linear(interaction_dim, interaction_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(interaction_dim // 2, n_tokens * out_dim),
        )

        self.role = nn.Parameter(torch.zeros(n_tokens, out_dim))
        nn.init.normal_(self.role, mean=0.0, std=0.02)

        self.norm = nn.LayerNorm(out_dim)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        """[B, D] x [B, D] -> [B, K, out_dim]."""
        diff = torch.abs(f1 - f2)
        prod = f1 * f2
        interaction = torch.cat([f1, f2, diff, prod], dim=-1)

        fused = self.proj(interaction)
        tokens = fused.view(-1, self.n_tokens, self.out_dim)
        tokens = tokens + self.role.unsqueeze(0)
        tokens = self.norm(tokens)
        return tokens


# =============================================================================
# 3. BinaryMatchHead
# =============================================================================

class BinaryMatchHead(nn.Module):
    """
    Mean+max pool over K tokens -> MLP -> 1 prob.
    """

    def __init__(
        self,
        in_dim: int = 512,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """[B, K, D] -> [B] raw logit (pre-sigmoid)."""
        mean_pool = tokens.mean(dim=1)
        max_pool, _ = tokens.max(dim=1)
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        logit = self.net(pooled).squeeze(-1)
        return torch.sigmoid(logit)


# =============================================================================
# 4. TransformationDecoder
# =============================================================================

class _SelfAttention(nn.Module):
    """
    Causal self-attention (learned pos-encoding mode).
    """

    def __init__(
        self,
        n_embd: int = 512,
        n_head: int = 8,
        dropout: float = 0.1,
        bias: bool = False,
        max_seq_len: int = 15,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=n_embd,
            num_heads=n_head,
            dropout=dropout,
            bias=bias,
            batch_first=True,
        )
        self.resid_dropout = nn.Dropout(dropout)
        mask = torch.triu(
            torch.full((max_seq_len, max_seq_len), float("-inf")), diagonal=1,
        )
        self.register_buffer("future_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T, _ = x.shape
        attn_mask = self.future_mask[:T, :T]
        q, k, v = x, x.clone(), x.clone()
        y, _ = self.attn(query=q, key=k, value=v,
                         attn_mask=attn_mask, need_weights=False)
        return self.resid_dropout(y)


class _CrossAttention(nn.Module):
    """
    Cross-attention from token stream to image context.
    """

    def __init__(
        self,
        n_embd: int = 512,
        n_head: int = 8,
        dropout: float = 0.1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=n_embd,
            num_heads=n_head,
            dropout=dropout,
            bias=bias,
            batch_first=True,
        )
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(query=x, key=ctx, value=ctx, need_weights=False)
        return self.resid_dropout(y)


class _MLP(nn.Module):
    """
    Feed-forward with GELU.
    """

    def __init__(
        self,
        n_embd: int = 512,
        ffn_ratio: int = 4,
        bias: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden = int(n_embd * ffn_ratio)
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden, bias=bias),
            nn.GELU(),
            nn.Linear(hidden, n_embd, bias=bias),
        )
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout_layer(self.net(x))


class _DecoderBlock(nn.Module):
    """
    Self-attn + cross-attn + FFN.
    """

    def __init__(
        self,
        n_embd: int = 512,
        n_head: int = 8,
        dropout: float = 0.1,
        bias: bool = False,
        max_seq_len: int = 15,
        ffn_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, elementwise_affine=bias)
        self.attn = _SelfAttention(n_embd, n_head, dropout, bias, max_seq_len)
        self.ln_2 = nn.LayerNorm(n_embd, elementwise_affine=bias)
        self.cross_attn = _CrossAttention(n_embd, n_head, dropout, bias)
        self.ln_3 = nn.LayerNorm(n_embd, elementwise_affine=bias)
        self.mlp = _MLP(n_embd, ffn_ratio, bias, dropout)

    def forward(
        self, x: torch.Tensor, images_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.cross_attn(self.ln_2(x), images_embeddings)
        x = x + self.mlp(self.ln_3(x))
        return x


class TransformationDecoder(nn.Module):
    """
    Autoregressive transformer decoder for transform sequences.
    """

    def __init__(
        self,
        n_embd: int = 512,
        n_head: int = 8,
        n_layer: int = 3,
        max_seq_len: int = 15,
        vocab_size: int = 13,
        dropout: float = 0.1,
        bias: bool = False,
        ffn_ratio: int = 4,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.n_embd = n_embd
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(max_seq_len, n_embd)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            _DecoderBlock(n_embd, n_head, dropout, bias, max_seq_len, ffn_ratio)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd, elementwise_affine=bias)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(
        self, images_embeddings: torch.Tensor, idx: torch.Tensor,
    ) -> torch.Tensor:
        """Forward returning last-position logits only.

        Args:
            images_embeddings: [B, K, D] fused context tokens.
            idx: [B, T] input token IDs.

        Returns:
            logits: [B, vocab_size] (last time-step).
        """
        _, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding(pos)
        x = self.dropout(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x, images_embeddings)
        x = self.ln_f(x)
        return self.lm_head(x)[:, -1, :]

    @torch.no_grad()
    def generate(
        self,
        images_embeddings: torch.Tensor,
        max_new_tokens: int = 10,
        temperature: float = 1.0,
        do_sample: bool = False,
    ) -> torch.Tensor:
        """
        Autoregressive greedy / sampled generation.

        Returns:
            token_ids: [B, 1 + max_new_tokens].
        """
        B = images_embeddings.shape[0]
        device = images_embeddings.device
        total_len = 1 + max_new_tokens

        idx = torch.full(
            (B, 1), self.bos_token_id, dtype=torch.long, device=device,
        )
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            next_logits = self.forward(images_embeddings, idx) / temperature

            next_logits[:, self.pad_token_id] = -float("inf")
            next_logits[:, self.bos_token_id] = -float("inf")

            if do_sample:
                probs = F.softmax(next_logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                idx_next = torch.argmax(next_logits, dim=-1)

            newly_finished = idx_next == self.eos_token_id
            finished = finished | newly_finished
            idx = torch.cat([idx, idx_next.unsqueeze(-1)], dim=1)

            if finished.all():
                break

        if idx.size(1) < total_len:
            pad = torch.full(
                (B, total_len - idx.size(1)),
                self.pad_token_id,
                device=device,
                dtype=torch.long,
            )
            idx = torch.cat([idx, pad], dim=1)
        else:
            idx = idx[:, :total_len]

        for i in range(B):
            eos_pos = (idx[i] == self.eos_token_id).nonzero(as_tuple=True)[0]
            if eos_pos.numel() > 0:
                idx[i, eos_pos[0].item() + 1 :] = self.pad_token_id

        return idx


# =============================================================================
# Full wrapper
# =============================================================================

class PlagiarismDetectionModelV2(nn.Module):
    """Full plagiarism model composed from the 4 standalone components.

    Three inference modes:
        1. ``forward()``            - p_eos similarity (backward-compat with V1)
        2. ``match_probability()``  - binary match-head probability
        3. ``generate_transform()`` - autoregressive sequence + match probability
    """

    def __init__(
        self, device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = ImageEncoder()
        self.fuser = ImageFuser()
        self.match_head = BinaryMatchHead()
        self.decoder = TransformationDecoder()

        self.to(self.device)

    # ----- Weight loading -----

    def load_components(
        self,
        encoder_path: str,
        fuser_path: str,
        match_head_path: str,
        decoder_path: str,
        strict: bool = True,
    ) -> None:
        """Load each component from individual state-dict files."""
        dev = self.device
        self.encoder.load_state_dict(
            torch.load(encoder_path, map_location=dev, weights_only=True),
            strict=strict,
        )
        self.fuser.load_state_dict(
            torch.load(fuser_path, map_location=dev, weights_only=True),
            strict=strict,
        )
        self.match_head.load_state_dict(
            torch.load(match_head_path, map_location=dev, weights_only=True),
            strict=strict,
        )
        self.decoder.load_state_dict(
            torch.load(decoder_path, map_location=dev, weights_only=True),
            strict=strict,
        )

    def load_from_full_checkpoint(
        self, checkpoint_path: str, strict: bool = True,
    ) -> None:
        """Load weights directly from a full training checkpoint.

        Handles both raw state-dicts and epoch-checkpoints
        (``{"model_state_dict": ...}``).
        """
        ck = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        sd = ck["model_state_dict"] if "model_state_dict" in ck else ck

        from export_plagiarism_weights import split_state_dict

        parts = split_state_dict(sd)
        self.encoder.load_state_dict(parts["encoder"], strict=strict)
        self.fuser.load_state_dict(parts["fuser"], strict=strict)
        self.match_head.load_state_dict(parts["match_head"], strict=strict)
        self.decoder.load_state_dict(parts["decoder"], strict=strict)
        print(f"[load] loaded all 4 components from {checkpoint_path}")

    # ----- Preprocessing -----

    def preprocess_batch(
        self,
        images: List[Union[bytes, Image.Image, np.ndarray, torch.Tensor]],
    ) -> torch.Tensor:
        """Convert heterogeneous inputs to [B, 3, 300, 300] tensor."""
        processed = []
        for img in images:
            if isinstance(img, bytes):
                img = Image.open(io.BytesIO(img))
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            elif isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img)
            processed.append(preprocess(img))
        return torch.stack(processed)

    # ----- Internal helpers -----

    def _encode_and_fuse(
        self,
        images1: torch.Tensor,
        images2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (emb1, emb2, fused_tokens)."""
        emb1 = self.encoder(images1.to(self.device))
        emb2 = self.encoder(images2.to(self.device))
        fused = self.fuser(emb1, emb2)
        return emb1, emb2, fused

    # ----- Mode 1: p_eos similarity (backward-compat) -----

    def forward(
        self,
        images1: torch.Tensor,
        images2: torch.Tensor,
    ) -> torch.Tensor:
        """Similarity = 1 - P(EOS | BOS, image_pair).

        Returns:
            similarities: [B] in (0, 1).
        """
        _, _, fused = self._encode_and_fuse(images1, images2)

        B = fused.shape[0]
        idx = torch.full(
            (B, 1), self.decoder.bos_token_id,
            dtype=torch.long, device=self.device,
        )

        logits = self.decoder(fused, idx)
        logits[:, self.decoder.pad_token_id] = -float("inf")
        logits[:, self.decoder.bos_token_id] = -float("inf")

        probs = torch.exp(logits) / torch.exp(logits).sum(dim=-1, keepdim=True)
        p_eos = probs[:, self.decoder.eos_token_id]

        return 1.0 - p_eos

    # ----- Mode 2: match-head probability -----

    @torch.inference_mode()
    def match_probability(
        self,
        images1: torch.Tensor,
        images2: torch.Tensor,
    ) -> torch.Tensor:
        """Binary match probability from the classification head.

        Returns:
            match_prob: [B] in (0, 1).
        """
        _, _, fused = self._encode_and_fuse(images1, images2)
        match_prob = self.match_head(fused)
        return match_prob

    # ----- Mode 3: full generation + match prob -----

    @torch.inference_mode()
    def generate_transform(
        self,
        images1: torch.Tensor,
        images2: torch.Tensor,
        max_new_tokens: int = 10,
        temperature: float = 1.0,
        do_sample: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive transform-sequence generation + match probability.

        Returns:
            token_ids: [B, 1 + max_new_tokens].
            match_prob: [B] in (0, 1).
        """
        _, _, fused = self._encode_and_fuse(images1, images2)
        match_prob = self.match_head(fused)
        token_ids = self.decoder.generate(
            fused,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )
        return token_ids, match_prob

    # ----- Batched similarity (production) -----

    @torch.inference_mode()
    def compute_similarities_batch(
        self,
        query_images: List[Union[bytes, np.ndarray, torch.Tensor, Image.Image]],
        candidate_images: List[List[Union[bytes, np.ndarray, torch.Tensor, Image.Image]]],
        mode: str = "decoder",
        candidate_chunk_size: int = 128,
        preprocess_batch_size: int = 64,
    ) -> List[List[float]]:
        """Batch similarity computation.

        Args:
            mode: ``"match_head"`` (recommended) or ``"p_eos"`` (backward-compat).
        """
        if not query_images:
            return []

        B = len(query_images)
        query_batch = self.preprocess_batch(query_images).to(self.device)
        query_embs = self.encoder(query_batch)
        results = []

        for i in range(B):
            candidates_i = candidate_images[i]
            if not candidates_i:
                results.append([])
                continue

            cand_embs_list = []
            for start in range(0, len(candidates_i), preprocess_batch_size):
                chunk = candidates_i[start : start + preprocess_batch_size]
                preprocessed = self.preprocess_batch(chunk).to(self.device)
                cand_embs_list.append(self.encoder(preprocessed))

            all_cand_embs = torch.cat(cand_embs_list, dim=0)
            N_cand = all_cand_embs.shape[0]
            similarities = []
            q_emb = query_embs[i : i + 1]

            for start in range(0, N_cand, candidate_chunk_size):
                end = min(start + candidate_chunk_size, N_cand)
                cand_chunk = all_cand_embs[start:end]
                C = cand_chunk.shape[0]
                q_expanded = q_emb.expand(C, -1)
                fused = self.fuser(cand_chunk, q_expanded)

                if mode == "match_head":
                    sims = self.match_head(fused)
                else:
                    idx = torch.full(
                        (C, 1), self.decoder.bos_token_id,
                        dtype=torch.long, device=self.device,
                    )
                    logits = self.decoder(fused, idx)
                    logits[:, self.decoder.pad_token_id] = -float("inf")
                    logits[:, self.decoder.bos_token_id] = -float("inf")
                    probs = torch.exp(logits) / torch.exp(logits).sum(dim=-1, keepdim=True)
                    p_eos = probs[:, self.decoder.eos_token_id]
                    sims = 1.0 - p_eos

                similarities.extend(sims.cpu().tolist())

            results.append(similarities)

        return results
