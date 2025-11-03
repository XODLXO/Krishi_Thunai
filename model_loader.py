# model_loader.py

import torch
import torch.nn as nn
import timm

# 1. Define the custom Hybrid Model Architecture
class HybridDenseInceptionSE(nn.Module):
    def __init__(self, num_classes=13, dense_pretrained=True, inception_pretrained=True, resse_pretrained=True):
        super(HybridDenseInceptionSE, self).__init__()

        # Feature extractor 1: DenseNet121
        densenet = timm.create_model('densenet121', pretrained=dense_pretrained, num_classes=0, global_pool='')
        self.densenet_features = nn.Sequential(*list(densenet.children())[:-1])
        dense_feature_dim = 1024 

        # Feature extractor 2: Inception_V4
        inception = timm.create_model('inception_v4', pretrained=inception_pretrained, num_classes=0, global_pool='avg')
        self.inception_features = inception
        inception_feature_dim = 1536 

        # Feature extractor 3: ResNeSt50d (or similar Res-SE block)
        resse = timm.create_model('resnest50d', pretrained=resse_pretrained, num_classes=0, global_pool='avg')
        self.resse_features = resse
        resse_feature_dim = 2048 
        
        # Combined dimension
        combined_dim = dense_feature_dim + inception_feature_dim + resse_feature_dim
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        # DenseNet feature extraction (needs pooling)
        dense_x = self.densenet_features(x)
        dense_x = nn.functional.adaptive_avg_pool2d(dense_x, 1).view(x.size(0), -1)

        # Inception and ResSE features (already pooled)
        inception_x = self.inception_features(x)
        resse_x = self.resse_features(x)

        # Concatenate
        combined_features = torch.cat((dense_x, inception_x, resse_x), dim=1)
        
        # Classification
        return self.classifier(combined_features)

# 2. Loading Function
def load_model(model_path, num_classes, device):
    """Loads the model state dictionary."""
    model = HybridDenseInceptionSE(num_classes=num_classes, dense_pretrained=False, inception_pretrained=False, resse_pretrained=False)
    
    # Load state dict, mapping to CPU if necessary for deployment
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.to(device)
    return model
