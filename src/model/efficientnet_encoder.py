import torch
import torch.nn as nn
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from omegaconf import DictConfig


class ImageEncoder(nn.Module):
    """
    Fixed-dimension image encoder based on pretrained EfficientNet-B3.

    This module extracts global image features using EfficientNet-B3 with the classifier head removed.
    It always outputs 1536-dimensional embeddings (the native output dimension of EfficientNet-B3).
    No projection layer is applied — the output dimension is fixed.

    Includes a built-in preprocessing pipeline compatible with EfficientNet.

    Attributes:
        preprocess (transforms.Compose): Standard preprocessing for EfficientNet inputs.
        feature_dim (int): Fixed output dimension (1536).

    Example:
        >>> encoder = ImageEncoder(freeze=True)
        >>> image_tensor = encoder.preprocess(pil_image)  # [3, 224, 224]
        >>> features = encoder(image_tensor.unsqueeze(0))  # [1, 1536]
    """

    def __init__(self, freeze=True):
        """
        Args:
            freeze (bool): If True, freezes all parameters of the EfficientNet backbone.
                           Defaults to True.
        """
        super().__init__()

        # Load pretrained EfficientNet-B3 and remove classification layers
        self.backbone = EfficientNet.from_pretrained('efficientnet-b3')
        self.backbone._dropout = nn.Identity()
        self.backbone._fc = nn.Identity()

        # Freeze or unfreeze backbone weights
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

        self.feature_dim = 1536  # Native output dimension of EfficientNet-B3

        # Built-in preprocessing pipeline
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def forward(self, x):
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 3, 224, 224].

        Returns:
            torch.Tensor: Feature embeddings of shape [batch_size, 1536].
        """
        return self.backbone(x)


class ImagePairEfficientNet(nn.Module):
    """
    Encoder for paired images that produces a fused embedding as a sequence of length 1.

    Processes two batches of images (or precomputed embeddings), extracts 1536-dim features
    from each using EfficientNet-B3, concatenates them into a 3072-dim vector,
    and projects the result into a learned combined embedding space of dimension `combined_emb_dim`.

    Always applies a linear fuser — there is no "raw concatenation" mode.

    The output is shaped as a sequence of length 1: [batch_size, 1, combined_emb_dim],
    which is suitable for downstream sequence-based models (e.g., transformers).

    Attributes:
        image_encoder (ImageEncoder): Shared encoder for both image inputs.
        fuser (nn.Linear): Projects concatenated features to `combined_emb_dim`.
        output_dim (int): Equals `combined_emb_dim` from config.
    """

    def __init__(self, config: DictConfig):
        """
        Args:
            config (DictConfig): Configuration object containing:
                - freeze_image_encoder (bool): Whether to freeze the image backbone.
                - combined_emb_dim (int): Target dimension for the fused output embedding.
        """
        super().__init__()
        self.config = config

        self.image_encoder = ImageEncoder(freeze=config.freeze_image_encoder)

        if not hasattr(config, 'combined_emb_dim'):
            raise ValueError("Config must contain 'combined_emb_dim'")

        concat_dim = self.image_encoder.feature_dim * 2  # 1536 * 2 = 3072
        self.fuser = nn.Linear(concat_dim, config.combined_emb_dim)
        self.output_dim = config.combined_emb_dim

    def extract_image_embeddings(self, image_batch_1, image_batch_2):
        """
        Extract 1536-dimensional embeddings from two image batches.

        Args:
            image_batch_1 (torch.Tensor): First batch of images, shape [B, 3, 224, 224].
            image_batch_2 (torch.Tensor): Second batch of images, shape [B, 3, 224, 224].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Two tensors of shape [B, 1536].
        """
        features_1 = self.image_encoder(image_batch_1)
        features_2 = self.image_encoder(image_batch_2)
        return features_1, features_2

    def forward(self, image_batch_1, image_batch_2, use_precomputed_embeddings=False):
        """
        Encode a pair of images (or embeddings) into a single fused token.

        Args:
            image_batch_1 (torch.Tensor): 
                Either raw images [B, 3, 224, 224] or precomputed embeddings [B, 1536].
            image_batch_2 (torch.Tensor): Same as image_batch_1.
            use_precomputed_embeddings (bool): 
                If True, inputs are treated as 1536-dim embeddings; 
                otherwise, they are passed through the image encoder.

        Returns:
            torch.Tensor: Fused embedding as a sequence of length 1, shape [B, 1, combined_emb_dim].
        """
        if not use_precomputed_embeddings:
            features_1, features_2 = self.extract_image_embeddings(image_batch_1, image_batch_2)
        else:
            features_1, features_2 = image_batch_1, image_batch_2

        concatenated = torch.cat([features_1, features_2], dim=1)  # [B, 3072]
        fused = self.fuser(concatenated)                           # [B, combined_emb_dim]
        return fused.unsqueeze(1)  # [B, 1, combined_emb_dim]