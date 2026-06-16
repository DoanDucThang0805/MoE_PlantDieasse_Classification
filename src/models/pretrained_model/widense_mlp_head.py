import timm
import torch
import torch.nn as nn
from torchinfo import summary

num_classes = 8

# remove classifier
model = timm.create_model(
    "mobilenetv3_small_100.lamb_in1k",
    pretrained=True,
    num_classes=0
)

# remove conv head 576 -> 1024
model.conv_head = nn.Identity()
model.act2 = nn.Identity()

# custom classifier
model.classifier = nn.Sequential(
    nn.Linear(576, 2048),
    nn.Hardswish(),
    nn.Dropout(0.2),

    nn.Linear(2048, 512),
    nn.Hardswish(),
    nn.Dropout(0.2),

    nn.Linear(512, num_classes)
)

summary(
    model,
    input_size=(1, 3, 224, 224),
    col_names=["input_size", "output_size", "num_params", "trainable"],
    col_width=20
)
