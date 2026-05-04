import torch
from torchinfo import summary
from thop import profile

from src.models.moe.model import MoEModel


dummt_input = torch.randn(1, 3, 224, 224)
context_input = torch.randn(1, 6)
model_config = {
    "context_dim": 6,
    "num_classes": 8,
    "num_experts": 4,
    "top_k": 4,
    "router_mode": "context_aware",
    "temperature": 0.5
}

model = MoEModel(**model_config)
summary(model, input_data=[dummt_input, context_input], col_names=["input_size", "output_size", "num_params", "trainable"])