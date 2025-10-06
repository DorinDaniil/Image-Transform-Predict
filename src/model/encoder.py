import torch
import torch.nn as nn
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from omegaconf import DictConfig


class ImageEncoder(nn.Module):
    """
    EfficientNet-B3 based image encoder that returns feature embeddings,
    optionally projected to a specified dimension.

    This encoder uses a pretrained EfficientNet-B3 model with the final classification layers removed.
    It includes built-in preprocessing for input images.

    Attributes:
        preprocess (transforms.Compose): Standard preprocessing pipeline for EfficientNet.
        feature_dim (int): Dimension of the output feature embeddings.

    Example:
        >>> encoder = ImageEncoder(encoder_emb_dim=768, freeze=True)
        >>> image_tensor = encoder.preprocess(pil_image)
        >>> features = encoder(image_tensor.unsqueeze(0))  # shape: [1, 768]

        >>> encoder = ImageEncoder(encoder_emb_dim=None)  # uses original 1536-dim features
    """

    def __init__(self, encoder_emb_dim=None, freeze=True):
        """
        Args:
            encoder_emb_dim (int or None): Target dimension for output embeddings.
                                           If None, uses original EfficientNet-B3 output (1536).
                                           If int, adds a projection layer to that dimension.
            freeze (bool): If True, freezes all backbone weights. Defaults to True.
        """
        super().__init__()

        # Load pretrained EfficientNet-B3 and remove classification layers
        self.backbone = EfficientNet.from_pretrained('efficientnet-b3')
        self.backbone._dropout = nn.Identity()
        self.backbone._fc = nn.Identity()

        # Set layer trainability
        self._set_trainable(freeze)

        self.original_feature_dim = 1536
        self.feature_dim = encoder_emb_dim if encoder_emb_dim is not None else self.original_feature_dim

        # Add projection layer only if a target dimension is specified and different from original
        if encoder_emb_dim is not None and encoder_emb_dim != self.original_feature_dim:
            self.projection = nn.Linear(self.original_feature_dim, encoder_emb_dim)
        else:
            self.projection = nn.Identity()

        # Built-in preprocessing pipeline
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _set_trainable(self, freeze):
        """Freeze or unfreeze all layers of the backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

    def forward(self, x):
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 3, 224, 224]

        Returns:
            torch.Tensor: Feature embeddings of shape [batch_size, feature_dim]
        """
        features = self.backbone(x)  # [batch_size, 1536]
        projected = self.projection(features)
        return projected


class ImagePairEncoder(nn.Module):
    """
    Encoder for computing similarity features between two sets of images.
    If `use_precomputed_embeddings` is True, expects precomputed embeddings as input.

    Behavior controlled by config:
        - use_fuser: if True, applies linear projection after concatenation.
                     if False, returns raw concatenated embeddings.
    """
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        # Initialize image encoder
        self.image_encoder = ImageEncoder(
            encoder_emb_dim=config.encoder_emb_dim,
            freeze=config.freeze_image_encoder
        )

        # Read fuser flag directly from config (required)
        self.use_fuser = config.use_fuser

        feature_dim = self.image_encoder.feature_dim
        concat_dim = feature_dim * 2

        if self.use_fuser:
            # Ensure combined_emb_dim is provided in config when fuser is used
            if not hasattr(config, 'combined_emb_dim'):
                raise ValueError("Config must contain 'combined_emb_dim' when use_fuser=True")
            self.fuser = nn.Linear(concat_dim, config.combined_emb_dim)
            self.output_dim = config.combined_emb_dim
        else:
            self.fuser = None
            self.output_dim = concat_dim

    def extract_image_embeddings(self, image_batch_1, image_batch_2):
        """Extract embeddings for two image batches."""
        features_1 = self.image_encoder(image_batch_1)
        features_2 = self.image_encoder(image_batch_2)
        return features_1, features_2

    def forward(self, image_batch_1, image_batch_2, use_precomputed_embeddings=False):
        """
        Args:
            image_batch_1, image_batch_2: either raw images [B, 3, 224, 224] or precomputed embeddings [B, D]
            use_precomputed_embeddings: whether inputs are already embeddings

        Returns:
            torch.Tensor: [B, output_dim]
                - If use_fuser=True: [B, combined_emb_dim]
                - If use_fuser=False: [B, 2 * encoder_emb_dim]
        """
        if not use_precomputed_embeddings:
            features_1, features_2 = self.extract_image_embeddings(image_batch_1, image_batch_2)
        else:
            features_1, features_2 = image_batch_1, image_batch_2

        concatenated = torch.cat([features_1, features_2], dim=1)

        if self.use_fuser:
            return self.fuser(concatenated)
        else:
            return concatenated