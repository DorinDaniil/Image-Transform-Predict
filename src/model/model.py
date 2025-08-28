import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import ImageEncoder
from .decoder import TransformDecoder, TRANSFORM_TOKENS


class ImageTransformPredictor(nn.Module):
    """
    Complete model for predicting transformation sequences between image pairs.
    Uses EfficientNet for image feature extraction and transformer for sequence prediction.
    """
    def __init__(self, image_feature_dim=1536, embedding_dim=256, num_heads=8, num_layers=6, max_seq_length=20):
        """
        Args:
            image_feature_dim: Dimension of image features from encoder
            embedding_dim: Dimension for token embeddings
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            max_seq_length: Maximum sequence length for the decoder
        """
        super().__init__()
        # Image encoder
        self.image_encoder = ImageEncoder(freeze=True)

        # Transformation decoder
        vocab_size = len(TRANSFORM_TOKENS)
        self.transform_decoder = TransformDecoder(
            image_feature_dim=image_feature_dim * 2,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            vocab_size=vocab_size,
            max_seq_length=max_seq_length
        )

    def forward(self, image1, image2, target_tokens=None):
        """
        Args:
            image1: First input image [batch_size, 3, 224, 224]
            image2: Second input image [batch_size, 3, 224, 224]
            target_tokens: Target transformation tokens [batch_size, seq_len] (optional)

        Returns:
            logits: Prediction logits [batch_size, seq_len, vocab_size]
            or generated sequence if target_tokens is None
        """
        # Get image features
        features1 = self.image_encoder(image1)
        features2 = self.image_encoder(image2)

        # Concatenate image features
        image_features = torch.cat([features1, features2], dim=1)

        # Predict transformation sequence
        return self.transform_decoder(image_features, target_tokens)
