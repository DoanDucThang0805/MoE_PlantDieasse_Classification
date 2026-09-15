import timm
from torchinfo import summary


model = timm.create_model(
    model_name='efficientnet_b0.ra_in1k',
    pretrained=True,
    num_classes=8
)
summary(
    model,
    input_size=(1, 3, 224, 224),
    col_names=["input_size", "output_size", "num_params", "mult_adds"],
)
