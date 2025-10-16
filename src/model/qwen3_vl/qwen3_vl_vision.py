# qwen3_vl_vision.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# --- Helper Functions ---

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding (RoPE) to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeats key/value heads to match the number of query heads (for GQA/MQA)."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# --- Activation Functions ---
ACT2FN = {
    "gelu_pytorch_tanh": lambda x: nn.GELU(approximate="tanh")(x),
    "silu": lambda x: x * torch.sigmoid(x),
}


# --- Vision Encoder Layers ---

class Qwen3VLVisionMLP(nn.Module):
    """MLP block used in Qwen3-VL vision transformer."""

    def __init__(self, config):
        super().__init__()
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class Qwen3VLVisionPatchEmbed(nn.Module):
    """Patch embedding using 3D convolution (supports video with T=1 for images)."""

    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.embed_dim = config.hidden_size
        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(3, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        x = self.proj(x)  # [B, D, T', H', W']
        return x.flatten(2).transpose(1, 2)  # [B, N, D]


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    """Rotary position embedding for vision tokens."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)


class Qwen3VLVisionPatchMerger(nn.Module):
    """Merges spatial patches and projects to language model embedding dimension."""

    def __init__(self, config, use_postshuffle_norm=False):
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm
        input_size = self.hidden_size if use_postshuffle_norm else config.hidden_size
        self.norm = nn.LayerNorm(input_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            x = x.view(-1, self.hidden_size)
        x = self.norm(x)
        if self.use_postshuffle_norm:
            x = x.view(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


class Qwen3VLVisionAttention(nn.Module):
    """Multi-head attention with RoPE for vision tokens."""

    def __init__(self, config):
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[0]
        qkv = self.qkv(hidden_states).reshape(seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(1)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos.unsqueeze(0), sin.unsqueeze(0))

        # Split by image/video segments using cu_seqlens
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        q_splits = torch.split(q, lengths.tolist(), dim=0)
        k_splits = torch.split(k, lengths.tolist(), dim=0)
        v_splits = torch.split(v, lengths.tolist(), dim=0)

        outputs = []
        for q_seg, k_seg, v_seg in zip(q_splits, k_splits, v_splits):
            q_seg = q_seg.transpose(0, 1).unsqueeze(0)  # [1, H, L, D]
            k_seg = k_seg.transpose(0, 1).unsqueeze(0)
            v_seg = v_seg.transpose(0, 1).unsqueeze(0)

            attn_weights = torch.matmul(q_seg, k_seg.transpose(-2, -1)) * self.scaling
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, v_seg)
            attn_output = attn_output.squeeze(0).transpose(0, 1).reshape(-1, self.dim)
            outputs.append(attn_output)

        output = torch.cat(outputs, dim=0)
        return self.proj(output)


class Qwen3VLVisionBlock(nn.Module):
    """Transformer block for vision encoder."""

    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(config)
        self.mlp = Qwen3VLVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens, position_embeddings
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


# --- Main Vision Encoder Model ---

class Qwen3VLVisionModel(nn.Module):
    """Vision encoder from Qwen3-VL, adapted for standalone use."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = Qwen3VLVisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings ** 0.5)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList([Qwen3VLVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=False)
        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList([
            Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=True)
            for _ in config.deepstack_visual_indexes
        ])

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        rotary_key = prefix + "rotary_pos_emb.inv_freq"
        if rotary_key not in state_dict:
            head_dim = self.config.hidden_size // self.config.num_heads
            dim = head_dim // 2
            theta = 10000.0
            inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
            self.rotary_pos_emb.register_buffer("inv_freq", inv_freq, persistent=False)
            if strict and rotary_key in missing_keys:
                missing_keys.remove(rotary_key)

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Computes rotary position embeddings based on grid layout."""
        merge_size = self.spatial_merge_size
        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device
        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)
        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size
            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]
            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)
            num_tokens = coords.shape[0]
            pos_ids[offset:offset + num_tokens] = coords
            offset += num_tokens
        embeddings = freq_table[pos_ids].flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Interpolates positional embeddings for variable-resolution inputs."""
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        device = grid_thw.device
        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]
        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)
            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor
            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side
            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())
        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])
        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        return torch.cat(patch_pos_embeds_permute)

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> Tuple[torch.Tensor, list]:
        """
        Forward pass of the vision encoder.

        Args:
            pixel_values: [B, C, T=1, H, W] — normalized image tensors.
            grid_thw: [B, 3] — [T, H_patches, W_patches] per image.

        Returns:
            visual_features: [N, out_hidden_size]
            deepstack_features: list of intermediate features (for DeepStack)
        """
        hidden_states = self.patch_embed(pixel_values)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        emb = torch.cat([rotary_pos_emb, rotary_pos_emb], dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_features = []
        for i, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, cu_seqlens, position_embeddings)
            if i in self.deepstack_visual_indexes:
                idx = self.deepstack_visual_indexes.index(i)
                deepstack_features.append(self.deepstack_merger_list[idx](hidden_states))

        hidden_states = self.merger(hidden_states)
        return hidden_states, deepstack_features