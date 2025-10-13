import torch
import torch.nn as nn
from typing import Tuple
from omegaconf import DictConfig


def interpolate_pos_encoding_2d(
    pos_embed: torch.Tensor,
    grid_h: int,
    grid_w: int,
    orig_grid_h: int = 14,
    orig_grid_w: int = 14,
) -> torch.Tensor:
    """Bicubic interpolation of 2D positional embeddings (DeiT / MAE style)."""
    if grid_h == orig_grid_h and grid_w == orig_grid_w:
        return pos_embed
    dim = pos_embed.shape[-1]
    pos_embed = pos_embed.reshape(1, orig_grid_h, orig_grid_w, dim).permute(0, 3, 1, 2)
    pos_embed = nn.functional.interpolate(
        pos_embed, size=(grid_h, grid_w), mode="bicubic", align_corners=False
    )
    return pos_embed.permute(0, 2, 3, 1).flatten(1, 2)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, n_embd: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(3, n_embd, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        B, _, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"Image dimensions must be divisible by patch_size={self.patch_size}"
        x = self.proj(x)
        grid_h, grid_w = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # [B, N, n_embd]
        return x, (grid_h, grid_w)


class FlexibleViTEncoder(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.n_embd = config.n_embd
        self.patch_size = config.patch_size
        self.use_cls_token = config.get("use_cls_token", False)

        self.patch_embed = PatchEmbed(patch_size=self.patch_size, n_embd=self.n_embd)
        self.pos_embed = nn.Parameter(torch.zeros(1, 14 * 14, self.n_embd))
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.n_embd))

        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(self.n_embd) if config.get("final_norm", True) else nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.n_embd,
            nhead=config.n_head,
            dim_feedforward=int(self.n_embd * config.ffn_ratio),
            dropout=config.dropout,
            activation=config.get("activation", "gelu"),
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layer, enable_nested_tensor=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, (grid_h, grid_w) = self.patch_embed(x)
        pos_embed = interpolate_pos_encoding_2d(self.pos_embed, grid_h, grid_w).to(x.device)

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            cls_pos = torch.zeros(1, 1, self.n_embd, device=x.device)
            pos_embed = torch.cat([cls_pos, pos_embed], dim=1)

        x = x + pos_embed
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.norm(x)
        return x


class ImagePairViT(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.encoder = FlexibleViTEncoder(config)
        self.sep_token = nn.Parameter(torch.randn(1, 1, config.n_embd))

    def extract_image_embeddings(self, image_batch_1, image_batch_2):
        """Extract embeddings for two image batches."""
        features_1 = self.encoder(image_batch_1)
        features_2 = self.encoder(image_batch_2)
        return features_1, features_2

    def forward(
        self,
        image_batch_1: torch.Tensor,
        image_batch_2: torch.Tensor,
        use_precomputed_embeddings: bool = False
    ) -> torch.Tensor:
        if not use_precomputed_embeddings:
            tokens1 = self.encoder(image_batch_1)
            tokens2 = self.encoder(image_batch_2)
        else:
            tokens1, tokens2 = image_batch_1, image_batch_2

        B = tokens1.shape[0]
        sep = self.sep_token.expand(B, -1, -1)
        return torch.cat([tokens1, sep, tokens2], dim=1)