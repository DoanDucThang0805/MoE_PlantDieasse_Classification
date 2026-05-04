import torch
from thop import profile
from models.pretrained_model.mobilenetv3_small import model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

x = torch.randn(1, 3, 224, 224).to(device)

flops, params = profile(
    model,
    inputs=(x,),
    verbose=True
)

print(f"Total FLOPs: {flops / 1e9:.3f} GFLOPs")
print(f"Total Params: {params}")
print(f"\nFLOPs per sample: {flops / 1 / 1e6:.2f} MFLOPs")
