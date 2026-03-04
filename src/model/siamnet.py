import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet


class SiamNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.extractor = EfficientNet.from_pretrained('efficientnet-b3')
        del self.extractor._fc
        
        self.dropf = nn.Dropout(p=0.20)
        
        self.fc1 = nn.Linear(1536, 1536)
        self.drop1 = nn.Dropout(p=0.20)
        self.fc2 = nn.Linear(1536, 1)
        self.sigmoid = nn.Sigmoid()
        
        self.preprocess = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        x = self.extractor._conv_stem(x)
        x = self.extractor._bn0(x)
        for layer in self.extractor._blocks:
            x = layer(x)
        x = self.extractor._conv_head(x)
        x = self.extractor._bn1(x)
        x = self.extractor._avg_pooling(x).view(b, -1)
        x = self.dropf(x)
        return x

    def head(self, diff: torch.Tensor) -> torch.Tensor:
        x = self.drop1(F.relu(self.fc1(diff)))
        x = self.sigmoid(self.fc2(x))
        return x.squeeze(-1)

    def forward(self, batch1: torch.Tensor, batch2: torch.Tensor) -> torch.Tensor:
        emb1 = self.encode(batch1)
        emb2 = self.encode(batch2)

        emb1_expanded = emb1.unsqueeze(1).expand(-1, emb2.size(0), -1)
        emb2_expanded = emb2.unsqueeze(0).expand(emb1.size(0), -1, -1)
        
        diff = torch.abs(emb1_expanded - emb2_expanded)
        diff_flat = diff.view(-1, 1536)
        
        logits_flat = self.head(diff_flat)
        return logits_flat.view(emb1.size(0), emb2.size(0))

    def predict_similarity(self, batch1: torch.Tensor, batch2: torch.Tensor) -> torch.Tensor:
        assert batch1.size(0) == batch2.size(0), "Batch sizes must match"
        
        emb1 = self.encode(batch1)
        emb2 = self.encode(batch2)
        
        diff = torch.abs(emb1 - emb2)
        return self.head(diff)

    def get_preprocessing(self):
        return self.preprocess