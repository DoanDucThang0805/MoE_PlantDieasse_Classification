"""
Quantization package for static INT8 quantization of pretrained models.

Submodules:
- quantize_utils: shared utilities (export, quantize, evaluate, compare)
- quantize_efficientnetb0: EfficientNet-B0 (timm)
- quantize_ghostnet: GhostNet (timm)
- quantize_mobilenetv3_small: MobileNetV3-Small (timm)
- quantize_mobilevits: MobileViT-S (timm)
- quantize_mobilevitxs: MobileViT-XS (timm)
- quantize_shufflenetv2: ShuffleNet-V2 x2.0 (torchvision)
- quantize_squeezenet: SqueezeNet 1.1 (torchvision)
- edge_benchmark_int8: Edge benchmark (size, latency, memory) for FP32 vs INT8
- eval_int8_accuracy: Accuracy & macro-F1 drop evaluation on PlantDoc/SLIF datasets
"""
