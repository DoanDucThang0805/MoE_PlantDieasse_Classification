import timm
from torchinfo import summary


num_classes=8
def create_model(num_classes):
    model = timm.create_model(
        'mobilenetv3_small_100',
        pretrained=True,
        num_classes=num_classes
    )
    return model



# Summary
summary(create_model(num_classes), (1,3,224,224), col_names=["input_size", "output_size", "num_params", "mult_adds"])
