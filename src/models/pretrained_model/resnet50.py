import timm
from torchinfo import summary


num_classes = 8
model = timm.create_model(
    "resnet50",
    pretrained=True,
    num_classes=num_classes
)

summary(
    model,
    input_size=(1, 3, 224, 224),
    col_names=["input_size", "output_size", "num_params", "mult_adds"],
)
