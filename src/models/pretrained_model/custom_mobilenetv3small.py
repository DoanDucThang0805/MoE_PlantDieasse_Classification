from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import torch.nn as nn
from torchinfo import summary


weights = MobileNet_V3_Small_Weights.DEFAULT
num_classes = 8

model = mobilenet_v3_small(weights=weights)

in_features = model.classifier[0].in_features   # 576


# widened dense classifier head
model.classifier = nn.Sequential(
    nn.Linear(in_features, 2048),
    nn.Hardswish(),
    nn.Dropout(0.2),
    nn.Linear(2048, 512),
    nn.Hardswish(),
    nn.Dropout(0.2),
    nn.Linear(512, num_classes)
)


summary(model, input_size=(1, 3, 224, 224), col_names=["input_size", "output_size", "num_params", "trainable"], col_width=20)
