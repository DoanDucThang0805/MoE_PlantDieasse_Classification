import torch
import torch.nn as nn
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights
from torchinfo import summary

# load pretrained
weights = SqueezeNet1_1_Weights.DEFAULT
model = squeezenet1_1(weights=weights)

# sửa output thành 8 class
model.classifier[1] = nn.Conv2d(512, 8, kernel_size=1)
model.num_classes = 8

summary(
    model,
    input_size=(1,3,224,224),
    col_names=["input_size","output_size","num_params","trainable"]
)