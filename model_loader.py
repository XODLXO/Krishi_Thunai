# backend/core/model_loader.py
import torch
from torchvision import models
import timm
from torch import nn

class ResidualSE(nn.Module):
    # ... (Your ResidualSE class definition) ...
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(1, channels // reduction))
        self.fc2 = nn.Linear(max(1, channels // reduction), channels)
        self.act = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.act(self.fc1(x))
        y = self.sigmoid(self.fc2(y))
        return x + x * y

class HybridDenseInceptionSE(nn.Module):
    # ... (Your HybridDenseInceptionSE class definition) ...
    def __init__(self, num_classes, densenet_pretrained=False, inception_pretrained=False, se_reduction=16, dropout=0.3):
        super().__init__()
        dnet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if densenet_pretrained else None)
        self.densenet_features = dnet.features
        densenet_dim = 1024

        inet = timm.create_model("inception_resnet_v2", pretrained=inception_pretrained, num_classes=1000)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 299, 299)
            feat = inet.forward_features(dummy)
            inception_dim = feat[-1].shape[1] if isinstance(feat, (list, tuple)) else feat.shape[1]

        self.inception = inet
        self.pool = nn.AdaptiveAvgPool2d(1)
        fused_dim = densenet_dim + inception_dim
        self.res_se = ResidualSE(fused_dim, se_reduction)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fused_dim, num_classes)

    def forward(self, x):
        d = self.pool(self.densenet_features(x)).flatten(1)
        i = self.pool(self.inception.forward_features(x)).flatten(1)
        f = self.res_se(torch.cat([d, i], 1))
        return self.classifier(self.dropout(f))

def load_model(weight_path, num_classes, device):
    # Uses the path passed from predictor.py
    model = HybridDenseInceptionSE(num_classes)
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model
