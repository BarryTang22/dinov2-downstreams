import torch
import torch.nn as nn
from transformers import AutoModel
import math

class ClassificationHead(nn.Module):
    def __init__(self, embedding_size=384, num_classes=5, hidden_size=512, dropout=0.1):
        super(ClassificationHead, self).__init__()
        
        self.head = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.head(x)

class dinov2_classifier(nn.Module):
    def __init__(self, embedding_size=384, num_classes=5, head_type='linear', 
                 hidden_size=512, head_path=None, pooling='cls', temperature=1.0):
        super(dinov2_classifier, self).__init__()
        self.dinov2 = AutoModel.from_pretrained('facebook/dinov2-with-registers-small')
        self.dinov2.eval()
        self.pooling = pooling
        self.temperature = temperature
        
        if head_path is not None:
            self.head = ClassificationHead(
                embedding_size=embedding_size,
                num_classes=num_classes,
                hidden_size=hidden_size
            )
            self.head.load_state_dict(torch.load(head_path))
        else:
            self.head = ClassificationHead(
                embedding_size=embedding_size,
                num_classes=num_classes,
                hidden_size=hidden_size
            )

    def forward(self, x):
        with torch.no_grad():
            outputs = self.dinov2(x)
            hidden_states = outputs.last_hidden_state  # [batch_size, num_patches, embedding_size]
            
            if self.pooling == 'cls':
                features = hidden_states[:, 0]
            else:
                features = hidden_states.mean(dim=1)
        
        logits = self.head(features)
        return logits / self.temperature

    def save_head(self, path):
        torch.save(self.head.state_dict(), path)

    def load_head(self, path):
        self.head.load_state_dict(torch.load(path))
        
    def extract_features(self, x):
        self.eval()
        with torch.no_grad():
            outputs = self.dinov2(x)
            hidden_states = outputs.last_hidden_state
            if self.pooling == 'cls':
                features = hidden_states[:, 0]
            else:
                features = hidden_states.mean(dim=1)
        return features
        
class SegmentationHead(nn.Module):
    def __init__(self, embedding_size=384, num_classes=150):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(embedding_size, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        return self.head(x)

class dinov2_segmenter(nn.Module):
    def __init__(self, embedding_size=384, num_classes=150):
        super().__init__()
        self.dinov2 = AutoModel.from_pretrained('facebook/dinov2-with-registers-small')
        self.dinov2.eval()
        
        self.head = SegmentationHead(
            embedding_size=embedding_size,
            num_classes=num_classes
        )
    
    def forward(self, x):
        B, C, H, W = x.shape  # [1, 3, 224, 224]
        
        with torch.no_grad():
            outputs = self.dinov2(x)
            features = outputs.last_hidden_state  # [B, 261, 384]
            # Remove CLS token
            features = features[:, 1:]  # [B, 260, 384]
            # The model actually uses 16x16 patches
            H_patches = W_patches = 16  # DINOv2 uses 16x16 grid
            embedding_dim = features.shape[-1]  # 384
            # Reshape to spatial dimensions
            features = features[:, :H_patches*W_patches].reshape(B, H_patches, W_patches, embedding_dim)
            features = features.permute(0, 3, 1, 2)  # [1, 384, 16, 16]
            
            # Upsample to input size
            features = nn.functional.interpolate(
                features, 
                size=(H, W),
                mode='bilinear', 
                align_corners=False
            )
        
        logits = self.head(features)
        return logits
    
    def save_head(self, path):
        torch.save(self.head.state_dict(), path)

    def load_head(self, path):
        self.head.load_state_dict(torch.load(path))
        
if __name__ == "__main__":
    model = dinov2_segmenter()
    x = torch.randn(1, 3, 224, 224)
    print(f"Input shape: {x.shape}")
    out = model(x)