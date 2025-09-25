import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchvision import transforms


class ImageEncoder(nn.Module):
    """
    EfficientNet-B3 based image encoder that returns 1536-dimensional feature embeddings.

    This encoder uses a pretrained EfficientNet-B3 model with the final classification layers removed.
    It includes built-in preprocessing for input images.

    Attributes:
        preprocess (transforms.Compose): Standard preprocessing pipeline for EfficientNet.
        feature_dim (int): Dimension of the output feature embeddings (1536 for EfficientNet-B3).

    Example:
        >>> encoder = EfficientNetB3Encoder(freeze=True)
        >>> image_tensor = encoder.preprocess(pil_image)
        >>> features = encoder(image_tensor.unsqueeze(0))
    """

    def __init__(self, freeze=True):
        """
        Args:
            freeze (bool): If True, freezes all backbone weights. Defaults to True.
        """
        super().__init__()

        # Load pretrained EfficientNet-B3 and remove classification layers
        self.backbone = EfficientNet.from_pretrained('efficientnet-b3')
        self.backbone._dropout = nn.Identity()
        self.backbone._fc = nn.Identity()

        # Set layer trainability
        self._set_trainable(freeze)
        self.feature_dim = 1536  # Output feature dimension for EfficientNet-B3

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

    def unfreeze_last_layers(self, n_layers=3):
        """
        Unfreeze the last N layers of the backbone for fine-tuning.

        Args:
            n_layers (int): Number of last layers to unfreeze. Defaults to 3.
        """
        # Freeze all layers first
        self._set_trainable(freeze=True)

        # Unfreeze final head layers
        for param in self.backbone._conv_head.parameters():
            param.requires_grad = True
        for param in self.backbone._bn1.parameters():
            param.requires_grad = True
        n_layers -= 1  # Count head as 1 layer

        # Unfreeze last MBConv blocks
        total_blocks = len(self.backbone._blocks)
        blocks_to_unfreeze = min(n_layers, total_blocks)

        for i in range(total_blocks - blocks_to_unfreeze, total_blocks):
            for param in self.backbone._blocks[i].parameters():
                param.requires_grad = True

    def forward(self, x):
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 3, 224, 224]

        Returns:
            torch.Tensor: Feature embeddings of shape [batch_size, 1536]
        """
        return self.backbone(x)


class ImagePairEncoder(nn.Module):
    """
    Encoder for computing similarity features between two sets of images.
    Uses EfficientNet-B3 to extract features from each image.

    Args:
        embedding_dim (int): Dimension of the output embeddings.
        freeze_image_encoder (bool): Whether to freeze the image encoder weights. Defaults to True.
        unfreeze_num_layers (int, optional): Number of last layers to unfreeze. Defaults to None.
    """

    def __init__(self, embedding_dim, freeze_image_encoder=True, unfreeze_n_layers=None):
        super().__init__()

        # Initialize image encoder
        self.image_encoder = ImageEncoder(freeze=freeze_image_encoder)

        # Unfreeze last layers if specified
        if unfreeze_n_layers is not None:
            self.image_encoder.unfreeze_last_layers(unfreeze_n_layers)

        # Initialize feature fuser
        self.fuser = nn.Linear(self.image_encoder.feature_dim * 2, embedding_dim)

    def forward(self, image_batch_1, image_batch_2):
        """
        Forward pass through the image pair encoder.

        Args:
            image_batch_1 (torch.Tensor): First batch of images with shape [batch_size_1, 3, 224, 224].
            image_batch_2 (torch.Tensor): Second batch of images with shape [batch_size_2, 3, 224, 224].

        Returns:
            torch.Tensor: Fused embeddings for pairs with shape [batch_size, embedding_dim].
        """
        # Extract features for both batches
        features_1 = self.image_encoder(image_batch_1)
        features_2 = self.image_encoder(image_batch_2)

        # Concatenate image features
        concatenated_features = torch.cat([features_1, features_2], dim=1)

        # Fuse features
        return self.fuser(concatenated_features)