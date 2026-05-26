import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet


class SiamNet(nn.Module):
    """
    Siamese network with an EfficientNet-B3 backbone for pairwise image similarity.

    The backbone uses the full EfficientNet forward pass (so all Swish activations
    and stochastic-depth scaling are applied as in the pretrained model). The
    final ``_dropout`` and ``_fc`` layers are replaced with ``nn.Identity``, so
    ``self.backbone(x)`` returns the 1536-dim global feature vector per image.

    A small comparison head (``|emb1 - emb2|`` -> MLP -> sigmoid) turns each pair
    into a similarity probability in ``[0, 1]``.
    """

    def __init__(self) -> None:
        super().__init__()

        self.backbone = EfficientNet.from_pretrained("efficientnet-b3")
        self.backbone._dropout = nn.Identity()
        self.backbone._fc = nn.Identity()

        self.feature_dim = 1536

        self.dropf = nn.Dropout(p=0.20)
        self.fc1 = nn.Linear(self.feature_dim, self.feature_dim)
        self.drop1 = nn.Dropout(p=0.20)
        self.fc2 = nn.Linear(self.feature_dim, 1)
        self.sigmoid = nn.Sigmoid()

        self.preprocess = transforms.Compose(
            [
                transforms.Resize((300, 300)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    # ------------------------------------------------------------------ #
    # Feature extraction
    # ------------------------------------------------------------------ #
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return a ``[B, 1536]`` global feature vector per image."""
        features = self.backbone(x)
        features = self.dropf(features)
        return features

    # ------------------------------------------------------------------ #
    # Comparison head
    # ------------------------------------------------------------------ #
    def head(self, diff: torch.Tensor) -> torch.Tensor:
        """Map a ``[*, 1536]`` absolute-difference vector to a probability ``[*]``."""
        x = self.drop1(F.relu(self.fc1(diff)))
        x = self.sigmoid(self.fc2(x))
        return x.squeeze(-1)

    # ------------------------------------------------------------------ #
    # Pairwise interfaces
    # ------------------------------------------------------------------ #
    def forward(self, batch1: torch.Tensor, batch2: torch.Tensor) -> torch.Tensor:
        """
        Build the full ``[B1, B2]`` similarity grid for two image batches.

        Args:
            batch1: ``[B1, 3, H, W]`` image tensor.
            batch2: ``[B2, 3, H, W]`` image tensor.

        Returns:
            ``[B1, B2]`` tensor of similarity probabilities.
        """
        emb1 = self.encode(batch1)
        emb2 = self.encode(batch2)

        emb1_expanded = emb1.unsqueeze(1).expand(-1, emb2.size(0), -1)
        emb2_expanded = emb2.unsqueeze(0).expand(emb1.size(0), -1, -1)

        diff = torch.abs(emb1_expanded - emb2_expanded)
        diff_flat = diff.view(-1, self.feature_dim)

        logits_flat = self.head(diff_flat)
        return logits_flat.view(emb1.size(0), emb2.size(0))

    def predict_similarity(
        self,
        batch1: torch.Tensor,
        batch2: torch.Tensor,
        use_precomputed_embeddings: bool = False,
    ) -> torch.Tensor:
        """
        Aligned (``i``-vs-``i``) similarity. Returns ``[B]`` probabilities.

        Args:
            batch1: ``[B, 3, H, W]`` images, or ``[B, 1536]`` precomputed features.
            batch2: same shape contract as ``batch1``.
            use_precomputed_embeddings: if ``True``, treat ``batch1``/``batch2``
                as already-encoded ``[B, 1536]`` feature tensors and skip the
                backbone forward pass. This is the path used by
                ``tuning_siamnet.py`` to avoid encoding each image twice.
        """
        assert batch1.size(0) == batch2.size(0), "Batch sizes must match"

        if use_precomputed_embeddings:
            emb1, emb2 = batch1, batch2
        else:
            emb1 = self.encode(batch1)
            emb2 = self.encode(batch2)

        diff = torch.abs(emb1 - emb2)
        return self.head(diff)

    def get_preprocessing(self):
        return self.preprocess
