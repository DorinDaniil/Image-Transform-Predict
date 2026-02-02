import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet


class SiamNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.extractor = EfficientNet.from_pretrained('efficientnet-b3')
        self.extractor._fc = nn.Identity()
        
        self.dropout_features = nn.Dropout(p=0.20)
        
        self.fc1 = nn.Linear(1536, 1536)
        self.dropout_head = nn.Dropout(p=0.20)
        self.fc2 = nn.Linear(1536, 1)
        self.sigmoid = nn.Sigmoid()
        
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extractor(x)
        return self.dropout_features(features)

    def head(self, diff: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(diff))
        x = self.dropout_head(x)
        x = self.fc2(x)
        return self.sigmoid(x).squeeze(-1)

    def forward(self, batch1: torch.Tensor, batch2: torch.Tensor) -> torch.Tensor:
        emb1 = self.encode(batch1)  # [B1, 1536]
        emb2 = self.encode(batch2)  # [B2, 1536]

        # Расширяем до [B1, B2, 1536] и вычисляем |emb1 - emb2|
        diff = torch.abs(emb1.unsqueeze(1) - emb2.unsqueeze(0))  # [B1, B2, 1536]
        diff_flat = diff.view(-1, 1536)  # [B1*B2, 1536]

        logits_flat = self.head(diff_flat)  # [B1*B2]
        return logits_flat.view(emb1.size(0), emb2.size(0))  # [B1, B2]

    def predict_similarity(self, batch1: torch.Tensor, batch2: torch.Tensor) -> torch.Tensor:
        assert batch1.size(0) == batch2.size(0), "Batch sizes must match for pairwise similarity"
        emb1 = self.encode(batch1)
        emb2 = self.encode(batch2)
        diff = torch.abs(emb1 - emb2)
        return self.head(diff)  # [B]

    def get_preprocessing(self):
        return self.preprocess