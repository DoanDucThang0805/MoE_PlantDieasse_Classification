import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from torchinfo import summary

weights = EfficientNet_V2_M_Weights.DEFAULT
model = efficientnet_v2_m(weights=weights)
num_classes = 8

# sửa classifier
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

summary(
    model,
    input_size=(1,3,224,224),
    col_names=["input_size","output_size","num_params","trainable"]
)