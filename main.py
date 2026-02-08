import timm
import torch

# List available MobileNetV4 variants (optional)
print([m for m in timm.list_models() if "mobilenetv4" in m])
