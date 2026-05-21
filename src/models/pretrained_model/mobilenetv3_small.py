import timm
from torchinfo import summary


num_classes=8
# Load model pretrained
model = timm.create_model(
    'mobilenetv3_small_100',
    pretrained=True,
    num_classes=num_classes
)


# Summary
summary(model, (1,3,224,224), col_names=["input_size", "output_size", "num_params", "mult_adds"])


# import torch
# from torchvision import models
# from torchinfo import summary

# model = models.mobilenet_v3_small(models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
# model.classifier[-1] = torch.nn.Linear(in_features=1024, out_features=8)
# summary(model, (1,3,224,224), col_names=["input_size", "output_size", "num_params", "mult_adds"])