import torch
from thop import profile
# from models.pretrained_model.widense_mlp_head import model
from models.pretrained_model.squeezenet import model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

x = torch.randn(1, 3, 224, 224).to(device)

flops, params = profile(
    model,
    inputs=(x,),
    verbose=True
)

print(f"Total FLOPs: {flops / 1e9:.4f} GFLOPs")
print(f"Total Params: {params}")
