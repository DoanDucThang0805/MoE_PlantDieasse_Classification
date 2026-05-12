"""
MobileNetV3 Small with Custom Classifier Head.

Backbone  : MobileNetV3-Small (pretrained ImageNet)
Head      : Linear → LayerNorm → GELU → Dropout → Linear → (classifier)
Author    : auto-generated
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights

try:
    from torchinfo import summary
except ImportError:
    summary = None


# ─────────────────────────────────────────────
#  Custom Classifier Head
# ─────────────────────────────────────────────
class CustomClassifierHead(nn.Module):
    """
    Projection head with classification layer.

    Architecture:
        Linear(embedding_dim → hidden_dim)
        LayerNorm(hidden_dim)
        GELU()
        Dropout(dropout)
        Linear(hidden_dim → embedding_dim)   ← projection / embedding output
        Linear(embedding_dim → classifier_input)
        LayerNorm(classifier_input)
        GELU()
        Dropout(dropout)
        Linear(classifier_input → num_classes)
    
    Args:
        embedding_dim: Input embedding dimension from backbone
        num_classes: Number of output classes
        dropout: Dropout rate (default: 0.1)
        hidden_dim: Hidden dimension in projection head (default: 1024)
        classifier_input: Hidden dimension in classifier (default: 256)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        hidden_dim: int = 1024,
        classifier_input: int = 256,
    ):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, classifier_input),
            nn.LayerNorm(classifier_input),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_input, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through projection and classification layers."""
        x = self.projection(x)   # (B, embedding_dim)
        x = self.classifier(x)   # (B, num_classes)
        return x


# ─────────────────────────────────────────────
#  MobileNetV3 Small Model
# ─────────────────────────────────────────────
class MobileNetV3SmallCustom(nn.Module):
    """
    MobileNetV3-Small with custom classifier head.

    Args:
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights (default: True)
        freeze_backbone: Freeze all backbone parameters (only train head)
        dropout: Dropout rate in projection head (default: 0.1)
        hidden_dim: Hidden dimension in projection head (default: 1024)
        classifier_input: Hidden dimension in classifier (default: 256)
    """

    # Output dimension of MobileNetV3-Small backbone after AdaptiveAvgPool
    EMBEDDING_DIM = 576

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.1,
        hidden_dim: int = 1024,
        classifier_input: int = 256,
    ):
        super().__init__()

        # Load backbone
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.mobilenet_v3_small(weights=weights)

        # Extract feature extraction layers and average pooling
        self.features = base.features  # Conv blocks
        self.avgpool = base.avgpool    # AdaptiveAvgPool2d(1, 1)

        # Custom classification head
        self.head = CustomClassifierHead(
            embedding_dim=self.EMBEDDING_DIM,
            num_classes=num_classes,
            dropout=dropout,
            hidden_dim=hidden_dim,
            classifier_input=classifier_input,
        )

        # Optionally freeze backbone parameters
        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters for transfer learning."""
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.avgpool.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone and head."""
        x = self.features(x)      # (B, 576, H', W')
        x = self.avgpool(x)       # (B, 576, 1, 1)
        x = torch.flatten(x, 1)   # (B, 576)
        x = self.head(x)          # (B, num_classes)
        return x



# ─────────────────────────────────────────────
#  Factory function
# ─────────────────────────────────────────────
def build_mobilenetv3_small(
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.1,
    hidden_dim: int = 1024,
    classifier_input: int = 256,
    verbose: bool = False,
) -> MobileNetV3SmallCustom:
    """
    Factory function to create and initialize MobileNetV3-Small model.
    
    Args:
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        freeze_backbone: Freeze backbone parameters
        dropout: Dropout rate
        hidden_dim: Hidden dimension in projection head
        classifier_input: Hidden dimension in classifier
        verbose: Print model summary if True
    
    Returns:
        MobileNetV3SmallCustom model instance
    """
    model = MobileNetV3SmallCustom(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
        hidden_dim=hidden_dim,
        classifier_input=classifier_input,
    )
    if verbose:
        print(model)
    return model

NUM_CLASSES = 8
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Create model
model = build_mobilenetv3_small(
    num_classes=NUM_CLASSES,
    pretrained=True,
    freeze_backbone=False,
    dropout=0.1,
    verbose=True,
)
model = model.to(DEVICE)
# ─────────────────────────────────────────────
#  Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if summary is not None:
        summary(
            model,
            (1, 3, 224, 224),
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            device=DEVICE,
        )

    # Test forward pass
    dummy = torch.randn(BATCH_SIZE, 3, 224, 224).to(DEVICE)
    with torch.no_grad():
        logits = model(dummy)
    
    print(f"\nInput shape:  {tuple(dummy.shape)}")
    print(f"Output shape: {tuple(logits.shape)}")
    print(f"Device:       {DEVICE}")
