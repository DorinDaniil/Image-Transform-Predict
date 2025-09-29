import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from omegaconf import DictConfig

from .tokenizer import VOCAB_SIZE, START_TOKEN_ID, END_TOKEN_ID, PAD_TOKEN_ID


class CombinedEmbeddingCrossAttention(nn.Module):
    """
    Cross-attention from token sequence to a single combined image embedding.
    Used to inject image context into the token stream.
    """
    def __init__(self, config: DictConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            dropout=config.dropout,
            bias=config.bias,
            batch_first=True,
        )
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        combined_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Token embeddings [B, T, n_embd].
            combined_embedding: Global image embedding [B, n_embd].
        
        Returns:
            Output of cross-attention [B, T, n_embd].
        """
        B, T, _ = x.shape
        k = v = combined_embedding.unsqueeze(1)  # [B, 1, n_embd]
        y, _ = self.attn(query=x, key=k, value=v, need_weights=False)
        y = self.resid_dropout(y)
        return y


class SelfAttention(nn.Module):
    """
    Causal self-attention (decoder-style) with built-in future masking.
    Ensures each token can only attend to itself and previous tokens.
    """
    def __init__(self, config: DictConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            dropout=config.dropout,
            bias=config.bias,
            batch_first=True,
        )
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal (future) mask: upper triangle = -inf
        mask = torch.triu(
            torch.full((config.max_seq_len, config.max_seq_len), float('-inf')),
            diagonal=1
        )
        self.register_buffer("future_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input token embeddings [B, T, n_embd].
        
        Returns:
            Output after causal self-attention [B, T, n_embd].
        """
        B, T, _ = x.shape
        attn_mask = self.future_mask[:T, :T].to(x.device)

        y, _ = self.attn(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask,
            need_weights=False,
        )
        y = self.resid_dropout(y)
        return y


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""
    def __init__(self, config: DictConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    """
    Transformer decoder block:
    - Causal self-attention
    - Cross-attention to image embedding
    - Feed-forward network
    """
    def __init__(self, config: DictConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)
        self.cross_attn = CombinedEmbeddingCrossAttention(config)
        self.ln_3 = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        combined_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Token embeddings [B, T, n_embd].
            combined_embedding: Image embedding [B, n_embd].
        
        Returns:
            Updated token embeddings [B, T, n_embd].
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.cross_attn(self.ln_2(x), combined_embedding)
        x = x + self.mlp(self.ln_3(x))
        return x


class TransformDecoder(nn.Module):
    """
    Autoregressive transformer decoder for generating token sequences conditioned on an image embedding.

    During training:
        - Input `idx` includes START, tokens, END, and padding up to `max_seq_len`.
        - `targets` is a left-shifted version of `idx` (i.e., what the model should predict at each step).
        - Padding tokens (PAD_TOKEN_ID) in `targets` are ignored in the loss.

    Example for a sequence [START, A, B, END] padded to length 6:
        idx     = [START, A, B, END, PAD, PAD]
        targets = [A, B, END, PAD, PAD, PAD]

    The loss ignores all positions where `targets == PAD_TOKEN_ID`, so the model only learns
    to predict real tokens (A, B, END) and never learns to predict PAD.
    """
    def __init__(self, config: DictConfig):
        """
        Initialize the decoder from a configuration object.

        Args:
            config (DictConfig): Must contain:
                - n_embd: int
                - n_head: int
                - n_layer: int
                - max_seq_len: int
                - vocab_size: int
                - dropout: float
                - bias: bool
        """
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        combined_embedding: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for training or inference.
        
        Args:
            idx: Input token IDs [B, T]. 
                 - Starts with START_TOKEN_ID.
                 - Contains real tokens, then END_TOKEN_ID, then PAD_TOKEN_ID up to max_seq_len.
            combined_embedding: Image context [B, n_embd].
            targets: Optional target token IDs [B, T] for training.
                     - Should be `idx` shifted left by one position.
                     - Last token is typically PAD_TOKEN_ID (no next token).
        
        Returns:
            logits: Predicted token logits [B, T, vocab_size].
            loss: Cross-entropy loss (None if targets not provided).
                  Padding positions (where targets == PAD_TOKEN_ID) are ignored.
        """
        B, T = idx.shape
        assert T <= self.config.max_seq_len, f"Input length {T} > max_seq_len {self.config.max_seq_len}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(pos)
        x = self.dropout(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x, combined_embedding)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Ignore padding tokens in loss computation
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=PAD_TOKEN_ID,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        combined_embedding: torch.Tensor,
        max_new_tokens: int = 10,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate sequences of length (1 + max_new_tokens).
        - Starts with START_TOKEN_ID.
        - Never generates PAD_TOKEN_ID.
        - All tokens after the first END_TOKEN_ID are replaced with PAD.
        """
        B = combined_embedding.shape[0]
        device = combined_embedding.device
        total_len = 1 + max_new_tokens

        if total_len > self.config.max_seq_len:
            raise ValueError(f"Total length {total_len} exceeds max_seq_len {self.config.max_seq_len}")

        idx = torch.full((B, 1), START_TOKEN_ID, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond, combined_embedding)
            next_logits = logits[:, -1, :] / temperature

            # Prevent PAD generation
            next_logits[:, PAD_TOKEN_ID] = -float('inf')
            next_logits[:, START_TOKEN_ID] = -float('inf')

            if top_k is not None:
                k = min(top_k, next_logits.size(-1))
                v, _ = torch.topk(next_logits, k)
                next_logits[next_logits < v[:, [-1]]] = -float('inf')

            probs = F.softmax(next_logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1).squeeze(-1)

            newly_finished = (idx_next == END_TOKEN_ID)
            finished = finished | newly_finished
            idx = torch.cat([idx, idx_next.unsqueeze(-1)], dim=1)

            if finished.all():
                break

        # Pad to total_len if needed
        if idx.size(1) < total_len:
            pad = torch.full((B, total_len - idx.size(1)), PAD_TOKEN_ID, device=device, dtype=torch.long)
            idx = torch.cat([idx, pad], dim=1)
        else:
            idx = idx[:, :total_len]

        # Replace everything after first END with PAD
        for i in range(B):
            end_pos = (idx[i] == END_TOKEN_ID).nonzero(as_tuple=True)[0]
            if end_pos.numel() > 0:
                first_end = end_pos[0].item()
                idx[i, first_end + 1:] = PAD_TOKEN_ID

        return idx