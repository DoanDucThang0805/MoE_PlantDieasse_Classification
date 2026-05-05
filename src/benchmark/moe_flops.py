import torch
from thop import profile
from models.moe.model import MoEModel


def complete_flops_analysis():
    # Model config
    config = {
        "context_dim": 6,
        "num_classes": 8,
        "num_experts": 4,
        "top_k": 2,
        "router_mode": "context_aware",
        "temperature": 0.5
    }
    
    model = MoEModel(**config).eval()
    
    # Input
    batch_size = 1
    x = torch.randn(batch_size, 3, 224, 224)
    context = torch.randn(batch_size, config["context_dim"])
    

    print("=" * 60)
    flops, params = profile(model, inputs=(x, context), verbose=True)
    print(f"Total FLOPs: {flops / 1e9:.3f} GFLOPs")
    print(f"Total Params: {params}")
    

if __name__ == "__main__":
    complete_flops_analysis()
