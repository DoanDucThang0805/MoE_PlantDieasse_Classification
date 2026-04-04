import torch
import torch.nn as nn
from torchvision import models
import timm


class Mobilenetv3SmallFeatureExtractor(nn.Module):
    """
    Extract feature vector from pretrained MobileNetV3-Small
    Output dim = 576
    """
    def __init__(self, pretrained=True, freeze_backbone=False) -> None:
        super().__init__()

        backbone = models.mobilenet_v3_small(pretrained=pretrained)

        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output_dim = 576

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x
    

class Mobilenetv3LargeFeatureExtractor(nn.Module):
    """
    Extract feature vector from pretrained MobileNetV3-Large
    Output dim = 960
    Input: [B, 3, 224, 224]
    Output: [B, 960]
    """
    def __init__(self, pretrained=True, freeze_backbone=False) -> None:
        super().__init__()

        backbone = models.mobilenet_v3_large(pretrained=pretrained)

        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output_dim = 960

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)      # [B, 960, 7, 7]
        x = self.pool(x)          # [B, 960, 1, 1]
        x = torch.flatten(x, 1)   # [B, 960]
        return x    


class VitB16FeatureExtractor(nn.Module):
    """
    Extract feature vector from pretrained ViT-B/16
    Output dim = 768
    """
    def __init__(self, pretrained=True, freeze_backbone=False) -> None:
        super().__init__()

        backbone = models.vit_b_16(pretrained=pretrained)

        # Remove classification head
        self.backbone = backbone
        self.backbone.heads = nn.Identity()

        self.output_dim = 768

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return x


class MobileViTV1FeatureExtractor(nn.Module):
    """
    MobileViT v1 - mobilevit_s
    Output dim ≈ 640
    """
    def __init__(self, pretrained=True, freeze_backbone=False):
        super().__init__()

        self.backbone = timm.create_model(
            "mobilevit_s",
            pretrained=pretrained,
            num_classes=0
        )

        self.output_dim = self.backbone.num_features

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.backbone(x)


class MobileViTV2FeatureExtractor(nn.Module):
    """
    MobileViT v2 - mobilevitv2_100
    Output dim ≈ 1024
    """
    def __init__(self, pretrained=True, freeze_backbone=False):
        super().__init__()

        self.backbone = timm.create_model(
            "mobilevitv2_100.cvnets_in1k",
            pretrained=pretrained,
            num_classes=0
        )

        self.output_dim = self.backbone.num_features

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.backbone(x)


class EfficientNetB4FeatureExtractor(nn.Module):

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


if __name__ == "__main__":
    dummy_input = torch.rand((1, 3, 224, 224))
    mobilenetv3smalloutput = Mobilenetv3SmallFeatureExtractor()(dummy_input)
    vitb16output = VitB16FeatureExtractor()(dummy_input)
    mobilevitv1 = MobileViTV1FeatureExtractor()(dummy_input)
    mobilevitv2 = MobileViTV2FeatureExtractor()(dummy_input)
    efficientnetb4 = EfficientNetB4FeatureExtractor()(dummy_input)

    print("mobilenetv3smalloutput.shape:", mobilenetv3smalloutput.shape)
    print("vitb16output.shape:", vitb16output.shape)
    print("mobilevitv1.shape:", mobilevitv1.shape)
    print("mobilevitv2.shape:", mobilevitv2.shape)
    print("efficientnetb4.shape:", efficientnetb4.shape)

    from torchinfo import summary
    summary(Mobilenetv3SmallFeatureExtractor(), (1,3,224,224))
    summary(Mobilenetv3LargeFeatureExtractor(), (1,3,224,224))
    summary(VitB16FeatureExtractor(), (1,3,224,224))
    summary(MobileViTV1FeatureExtractor(), (1,3,224,224))
    summary(MobileViTV2FeatureExtractor(), (1,3,224,224))
    summary(EfficientNetB4FeatureExtractor(), (1,3,224,224))
