class FeatureFuser(nn.Module):
    """
    Feature fusion module that combines difference and product embeddings of image pairs.

    This module takes pre-computed difference and element-wise product features of image pairs
    and combines them into a single fused representation.

    Args:
        feature_dim (int): Dimension of input feature vectors
        dropout_rate (float): Dropout probability. Defaults to 0.1.
    """

    def __init__(self, feature_dim, dropout_rate=0.1):
        super().__init__()

        # Pathway for processing absolute difference features
        self.diff_pathway = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Pathway for processing element-wise product features
        self.product_pathway = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

        # Final fusion layer
        self.fusion = nn.Linear(2 * feature_dim, feature_dim)

    def forward(self, diff_embeddings, product_embeddings):
        """
        Forward pass with pre-computed difference and product features.

        Args:
            diff_embeddings (torch.Tensor): Tensor of shape [batch_size, feature_dim]
                representing absolute differences between embeddings
            product_embeddings (torch.Tensor): Tensor of shape [batch_size, feature_dim]
                representing element-wise products of embeddings

        Returns:
            torch.Tensor: Fused embeddings of shape [batch_size, feature_dim]
        """
        diff_features = self.diff_pathway(diff_embeddings)
        product_features = self.product_pathway(product_embeddings)

        # Concatenate and fuse both representations
        combined = torch.cat([diff_features, product_features], dim=1)
        return self.fusion(combined)


class ImagePairEncoder(nn.Module):
    """
    Image pair encoder that computes similarity features between two sets of images.

    This encoder uses EfficientNet-B3 to extract features from each image, then computes
    difference and product features between all pairs of images, and finally fuses
    these features into a single representation.

    Args:
        freeze_image_encoder (bool): Whether to freeze the image encoder weights. Defaults to True.
        unfreeze_n_layers (int, optional): Number of last layers to unfreeze. Defaults to None.
    """

    def __init__(self, freeze_image_encoder=True, unfreeze_n_layers=None):
        super().__init__()

        # Initialize image encoder
        self.image_encoder = ImageEncoder(freeze=freeze_image_encoder)

        # Unfreeze last layers if specified
        if unfreeze_n_layers is not None:
            self.image_encoder.unfreeze_last_layers(unfreeze_n_layers)

        # Initialize feature fuser
        self.fuser = FeatureFuser(self.image_encoder.feature_dim)

    def forward(self, batch1, batch2):
        """
        Forward pass through the image pair encoder.

        Args:
            batch1 (torch.Tensor): First batch of images with shape [batch_size1, 3, 224, 224]
            batch2 (torch.Tensor): Second batch of images with shape [batch_size2, 3, 224, 224]

        Returns:
            torch.Tensor: Fused embeddings for all pairs with shape [batch_size1 * batch_size2, feature_dim]
        """
        # Extract features for both batches
        emb1 = self.image_encoder(batch1)
        emb2 = self.image_encoder(batch2)

        batch_size1, batch_size2 = emb1.size(0), emb2.size(0)

        # Expand embeddings for pairwise computation
        emb1_expanded = emb1.unsqueeze(1).expand(-1, batch_size2, -1)
        emb2_expanded = emb2.unsqueeze(0).expand(batch_size1, -1, -1)

        # Compute difference and product features
        diff_embeddings = torch.abs(emb1_expanded - emb2_expanded)
        product_embeddings = emb1_expanded * emb2_expanded

        # Reshape for fuser
        diff_embeddings = diff_embeddings.view(-1, self.image_encoder.feature_dim)
        product_embeddings = product_embeddings.view(-1, self.image_encoder.feature_dim)

        # Fuse features
        return self.fuser(diff_embeddings, product_embeddings)