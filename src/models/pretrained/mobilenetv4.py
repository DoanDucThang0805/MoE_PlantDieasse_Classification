import timm
from torchinfo import summary


num_classes=8
# Load model pretrained
model = timm.create_model(
    'mobilenetv4_conv_small.e1200_r224_in1k',
    pretrained=True,
    num_classes=num_classes
)


# Summary
summary(model, (1,3,224,224), col_names=["input_size", "output_size", "num_params", "mult_adds"])
