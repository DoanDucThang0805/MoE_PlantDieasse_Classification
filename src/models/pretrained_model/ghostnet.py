import timm
from torchinfo import summary


model = timm.create_model('ghostnet_100.in1k', pretrained=True)
summary(model, (1,3,224,224), col_names=["input_size", "output_size", "num_params", "mult_adds"])
