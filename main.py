import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights


class EfficientNetV2MBackbone(nn.Module):
    """
    EfficientNetV2-M backbone for feature extraction
    Output: feature vector [B, 1280]
    """

    def __init__(self, pretrained=True, freeze_backbone=False):
        super().__init__()

        weights = EfficientNet_V2_M_Weights.DEFAULT if pretrained else None

        model = efficientnet_v2_m(weights=weights)

        # backbone CNN
        self.features = model.features

        # global pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

        # embedding dimension
        self.feature_dim = model.classifier[1].in_features

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        return x
    
model = EfficientNetV2MBackbone()

x = torch.randn(2,3,224,224)

feat = model(x)

print(feat.shape)
print(torch.cuda.is_available())