import torch
from transformers import MobileViTForImageClassification
from torchinfo import summary


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small")
summary(model, input_size=(1, 3, 224, 224), col_names=["input_size", "output_size", "num_params", "mult_adds"])
