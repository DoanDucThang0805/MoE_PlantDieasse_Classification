"""Capacity-matched control for the MoE classifier.

Same MobileNetV3-Small feature extractor, same four 576->1024->576 experts,
same residual path and same final classifier as the proposed model, but NO
input-dependent router. All expert outputs are averaged uniformly.

The only removed parameters are the small context-aware gate (~8.6k params,
~0.15% of the full current model), making this a much cleaner parameter-capacity
control than the older 3.16M / 3.33M baselines.
"""
import torch
import torch.nn as nn
from models.moe.backbone import Mobilenetv3SmallFeatureExtractor


class StaticUniformExpertModel(nn.Module):
    def __init__(self, num_classes=8, num_experts=4, model_dim=576, expert_hidden=1024, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.feature_extractor = Mobilenetv3SmallFeatureExtractor(pretrained=pretrained, freeze_backbone=False)
        model_dim = self.feature_extractor.output_dim
        self.pre_moe_norm = nn.LayerNorm(model_dim)
        self.post_moe_norm = nn.LayerNorm(model_dim)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(model_dim, expert_hidden),
                nn.LayerNorm(expert_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(expert_hidden, model_dim),
            ) for _ in range(num_experts)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        feature = self.feature_extractor(x)
        z = self.pre_moe_norm(feature)
        expert_stack = torch.stack([expert(z) for expert in self.experts], dim=1)
        expert_mean = expert_stack.mean(dim=1)
        z = self.post_moe_norm(feature + expert_mean)
        return self.classifier(z)
