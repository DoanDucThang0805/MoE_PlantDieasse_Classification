import torchvision.models as models
import torch.nn as nn


class EfficientNetB4_Backbone(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()

        model = models.efficientnet_b4(pretrained=pretrained)

        self.features = model.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.out_dim = 1792

    def forward(self, x):

        x = self.features(x)
        x = self.pool(x)

        x = x.flatten(1)

        return x