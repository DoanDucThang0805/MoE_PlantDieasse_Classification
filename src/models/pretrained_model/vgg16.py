import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
from torchinfo import summary

# load pretrained
weights = VGG16_Weights.DEFAULT
model = vgg16(weights=weights)

num_classes = 8

# sửa classifier cuối
model.classifier[6] = nn.Linear(4096, num_classes)

# xem architecture
summary(model, input_size=(1, 3, 224, 224),
        col_names=["input_size", "output_size", "num_params", "trainable"])
