import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class StandardTopKgating(nn.Module):
    def __init__(self, model_dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.model_dim = model_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate_projector = nn.Linear(self.model_dim, self.num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_scores = self.gate_projector(x)
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        top_k_gates = torch.softmax(top_k_scores, dim=-1)
        

class NoisyTopKGating(nn.Module):
    def __init__(self, model_dim: int, num_experts: int, top_k: int, noise_stddev=1.0):
        super().__init__()
        self.model_dim = model_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_stddev = noise_stddev

        self.gate_projector = nn.Linear(model_dim, num_experts, bias=False)
        self.noise_layer = nn.Linear(model_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor):

        clean_logits = self.gate_projector(x)

        if self.training:
            noise_magnitude = self.noise_layer(x)
            noise_scale = F.softplus(noise_magnitude)

            # Gaussian noise
            sampled_noise = torch.randn_like(clean_logits)

            noisy_logits = clean_logits + noise_scale * sampled_noise * self.noise_stddev
        else:
            noisy_logits = clean_logits

        top_k_logits, top_k_indices = torch.topk(noisy_logits, self.top_k, dim=-1)

        combined_weights = F.softmax(top_k_logits, dim=-1)

        return combined_weights, top_k_indices, clean_logits    

if __name__ == "__main__":
    noisygating = NoisyTopKGating(
        model_dim=960,
        num_experts=4,
        top_k=3
    )
    logits = torch.rand((3, 960))
    combined_weights, top_k_indices, clean_logits = noisygating(logits)
    print(combined_weights)
    print(top_k_indices)
    