import timm
from torchinfo import summary


model = timm.create_model(
    model_name="mobilevit_xxs.cvnets_in1k",
    pretrained=True,
    num_classes=8
)
summary(model, (1,3,224,224), col_names=["input_size", "output_size", "num_params", "mult_adds"])
