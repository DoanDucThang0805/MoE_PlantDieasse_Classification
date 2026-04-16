import timm
from torchinfo import summary


model = timm.create_model('mobilevitv2_100.cvnets_in1k', pretrained=True)
summary(model, input_size=(1, 3, 224, 224), col_names=["input_size", "output_size", "num_params", "mult_adds"])
