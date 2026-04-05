import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchinfo import summary

weights = EfficientNet_V2_S_Weights.DEFAULT
model = efficientnet_v2_s(weights=weights)
num_classes = 8

# sửa classifier thành 8 class
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

summary(
    model,
    input_size=(1,3,224,224),
    col_names=["input_size","output_size","num_params","trainable"]
)