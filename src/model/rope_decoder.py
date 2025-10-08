import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from omegaconf import DictConfig


# ---------- RoPE utilities ----------
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding to query and key.
    Args:
        q, k: [B, T, n_embd] or [B, n_head, T, head_dim]
        cos, sin: [1, T, head_dim] or [T, head_dim]
    """
    # Ensure cos/sin have same number of dims as q/k
    while len(cos.shape) < len(q.shape):
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def get_rotary_embeddings(seq_len: int, dim: int, device: torch.device, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cos and sin for RoPE for a given sequence length.
    """
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(t, inv_freq)  # [seq_len, dim//2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim]
    cos = emb.cos()
    sin = emb.sin()
    return cos, sin


# ---------- Cross-attention ----------
class CombinedEmbeddingCrossAttention(nn.Module):
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

    def forward(self, x: torch.Tensor, combined_embedding: torch.Tensor) -> torch.Tensor:
        k = v = combined_embedding.unsqueeze(1)  # [B, 1, n_embd]
        y, _ = self.attn(query=x, key=k, value=v, need_weights=False)
        y = self.resid_dropout(y)
        return y


# ---------- RoPE Self-Attention ----------
class RoPESelfAttention(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask
        mask = torch.triu(
            torch.full((config.max_seq_len, config.max_seq_len), float('-inf')),
            diagonal=1
        )
        self.register_buffer("future_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        device = x.device

        # Project to Q, K, V
        q = self.q_proj(x)  # [B, T, C]
        k = self.k_proj(x)  # [B, T, C]
        v = self.v_proj(x)  # [B, T, C]

        # Reshape for multi-head: [B, T, n_head, head_dim] -> [B, n_head, T, head_dim]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        cos, sin = get_rotary_embeddings(T, self.head_dim, device, dtype=x.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product attention with causal mask
        attn_mask = self.future_mask[:T, :T]
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False  # we provide explicit mask
        )  # [B, n_head, T, head_dim]

        # Merge heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        y = self.resid_dropout(y)
        return y


# ---------- MLP ----------
class MLP(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, config.n_hidden_state, bias=config.bias),
            nn.GELU(),
            nn.Linear(config.n_hidden_state, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)
        self.attn = RoPESelfAttention(config)  # <-- RoPE instead of SelfAttention
        self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)
        self.cross_attn = CombinedEmbeddingCrossAttention(config)
        self.ln_3 = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, combined_embedding: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.cross_attn(self.ln_2(x), combined_embedding)
        x = x + self.mlp(self.ln_3(x))
        return x


# ---------- Main Decoder with RoPE ----------
class TransformDecoder(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.pad_token_id = config.pad_token_id

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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, T = idx.shape
        assert T <= self.config.max_seq_len, f"Input length {T} > max_seq_len {self.config.max_seq_len}"

        tok_emb = self.token_embedding(idx)  # [B, T, n_embd]
        x = self.dropout(tok_emb)  # <-- NO pos_emb added

        for block in self.blocks:
            x = block(x, combined_embedding)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=self.pad_token_id,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        combined_embedding: torch.Tensor,
        max_new_tokens: int = 10,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        do_sample: bool = False,
        pad_token_id: int = None,
        bos_token_id: int = None,
        eos_token_id: int = None,
    ) -> torch.Tensor:
        B = combined_embedding.shape[0]
        device = combined_embedding.device
        total_len = 1 + max_new_tokens
        if total_len > self.config.max_seq_len:
            raise ValueError(f"Total length {total_len} exceeds max_seq_len {self.config.max_seq_len}")

        if pad_token_id is None:
            pad_token_id = self.pad_token_id
        if bos_token_id is None:
            bos_token_id = self.bos_token_id
        if eos_token_id is None:
            eos_token_id = self.eos_token_id

        idx = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond, combined_embedding)
            next_logits = logits[:, -1, :] / temperature

            next_logits[:, pad_token_id] = -float('inf')
            next_logits[:, bos_token_id] = -float('inf')

            if top_k is not None:
                k = min(top_k, next_logits.size(-1))
                v, _ = torch.topk(next_logits, k)
                next_logits[next_logits < v[:, [-1]]] = -float('inf')

            if not do_sample:
                idx_next = torch.argmax(next_logits, dim=-1)
            else:
                probs = F.softmax(next_logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1).squeeze(-1)

            newly_finished = (idx_next == eos_token_id)
            finished = finished | newly_finished
            idx = torch.cat([idx, idx_next.unsqueeze(-1)], dim=1)

            if finished.all():
                break

        if idx.size(1) < total_len:
            pad = torch.full((B, total_len - idx.size(1)), pad_token_id, device=device, dtype=torch.long)
            idx = torch.cat([idx, pad], dim=1)
        else:
            idx = idx[:, :total_len]

        for i in range(B):
            end_pos = (idx[i] == eos_token_id).nonzero(as_tuple=True)[0]
            if end_pos.numel() > 0:
                first_end = end_pos[0].item()
                idx[i, first_end + 1:] = pad_token_id

        return idx