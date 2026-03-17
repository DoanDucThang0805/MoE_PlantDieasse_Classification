import torch.nn as nn
from torchvision import models
from torchinfo import summary


num_classes = 8
# Load pretrained
model = models.mobilenet_v3_large(pretrained=True)

# Thay output layer
model.classifier[3] = nn.Linear(
    in_features=1280,
    out_features=num_classes
)

summary(model, (1,3,224,224), col_names=["input_size", "output_size", "num_params", "mult_adds"])
