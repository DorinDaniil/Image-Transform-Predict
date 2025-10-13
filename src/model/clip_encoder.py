import torch
import torch.nn as nn
import open_clip
from torchvision.transforms import Compose
from omegaconf import DictConfig


class CLIPViTEncoder(nn.Module):
    """
    Minimal CLIP ViT encoder with built-in preprocessing.
    
    - Uses official CLIP vision encoder (no custom patches/pos embeds).
    - Includes CLIP's official preprocessing pipeline.
    - Projects global image embedding to decoder's n_embd space.
    """
    def __init__(self, config: DictConfig):
        super().__init__()
        # Load model and its official preprocessing
        self.clip_model, self.preprocess, _ = open_clip.create_model_and_transforms(
            config.clip_model_name,
            pretrained=config.pretrained,
            device="cpu"
        )
        self.vit = self.clip_model.visual

        # Freeze backbone if needed
        freeze = config.get("freeze", True)
        for param in self.vit.parameters():
            param.requires_grad = not freeze

        # Projection to decoder's embedding dimension
        self.proj = nn.Linear(self.vit.output_dim, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images [B, 3, H, W] — already normalized with CLIP's stats.
        Returns:
            embeddings: [B, n_embd]
        """
        clip_features = self.vit(x)  # [B, D_clip]
        return self.proj(clip_features)  # [B, n_embd]
    

class ImagePairViT(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.encoder = CLIPViTEncoder(config)

    @property
    def preprocess(self) -> Compose:
        """Expose CLIP's official preprocessing pipeline."""
        return self.encoder.preprocess

    def forward(
        self,
        image_batch_1: torch.Tensor,
        image_batch_2: torch.Tensor,
        use_precomputed_embeddings: bool = False
    ) -> torch.Tensor:
        """
        Returns:
            Combined image context as [B, 2, n_embd]
        """
        if not use_precomputed_embeddings:
            emb1 = self.encoder(image_batch_1)
            emb2 = self.encoder(image_batch_2)
        else:
            emb1, emb2 = image_batch_1, image_batch_2

        return torch.stack([emb1, emb2], dim=1)  # [B, 2, n_embd]
    

config = DictConfig({
    "clip_model_name": "ViT-B-16",
    "pretrained": "laion2b_s34b_b88k",
    "freeze": True,
    "n_embd": 768
})