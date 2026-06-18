import torch
from thop import profile
# from models.pretrained_model.widense_mlp_head import model
# from models.dense_multibranch.mobilenetv3_small_dense_multibranch import MobileNetV3SmallDenseMultiBranch
from models.pretrained_model.squeezenet import model
# model = MobileNetV3SmallDenseMultiBranch(num_classes=8, num_experts=4)

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
