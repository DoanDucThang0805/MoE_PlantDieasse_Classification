import torch.nn as nn
from torchinfo import summary
from torchvision import models


num_classes = 8

try:
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
except AttributeError:
    # Fallback for older torchvision versions.
    model = models.efficientnet_b0(pretrained=True)

# Replace classification head for the target number of classes.
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(
    in_features=in_features,
    out_features=num_classes,
)

summary(
    model,
    input_size=(1, 3, 224, 224),
    col_names=["input_size", "output_size", "num_params", "mult_adds"],
)
