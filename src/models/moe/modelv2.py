"""
ONNX-friendly Mixture of Experts (MoE) model.

This module keeps the same public model interface as model.py, but rewrites the
expert routing step without data-dependent Python control flow, boolean indexing,
or in-place masked assignment. That makes the forward graph suitable for ONNX
export in eval/inference mode.
"""

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn

from .backbone import Mobilenetv3SmallFeatureExtractor
from .gating import NoisyTopKGating, ContextAwareLinearGating, ContextAwareGating


class MoELayer(nn.Module):
    """
    ONNX-friendly MoE layer.

    Equivalent to the MoELayer in model.py for inference:
    - gating still selects top-k experts
    - selected experts are weighted by router probabilities
    - non-selected experts receive zero weight

    Difference:
    - all experts are evaluated for every sample, then combined by router weights.
      This avoids dynamic boolean indexing and masked in-place assignment.
    """

    def __init__(
        self,
        context_dim: Optional[int],
        model_dim: int,
        num_experts: int,
        top_k: int,
        router_mode: Literal["noisy", "context_aware"],
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_mode = router_mode
        self.temperature = temperature

        if not (0 < self.top_k <= self.num_experts):
            raise ValueError(
                "top_k must be a positive integer less than or equal to num_experts"
            )

        if self.router_mode == "noisy":
            self.gating = NoisyTopKGating(
                model_dim=model_dim,
                num_experts=self.num_experts,
                top_k=self.top_k,
                temperature=self.temperature,
            )
        elif self.router_mode == "context_aware":
            self.gating = ContextAwareGating(
                model_dim=model_dim,
                context_dim=context_dim,
                num_experts=self.num_experts,
                top_k=self.top_k,
                temperature=self.temperature,
            )
        else:
            raise ValueError(
                f"Invalid router_mode: {self.router_mode}. "
                "Must be 'noisy' or 'context_aware'."
            )

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(model_dim, 1024),
                    nn.LayerNorm(1024),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(1024, model_dim),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.router_mode == "noisy":
            combined_weights, top_k_indices, clean_router_logits = self.gating(x)
        elif self.router_mode == "context_aware":
            combined_weights, top_k_indices, clean_router_logits = self.gating(
                x,
                context,
            )
        else:
            raise ValueError(
                f"Invalid router_mode: {self.router_mode}. "
                "Must be 'noisy' or 'context_aware'."
            )

        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts],
            dim=1,
        )

        router_weights = torch.zeros(
            x.size(0),
            self.num_experts,
            dtype=x.dtype,
            device=x.device,
        )
        router_weights = router_weights.scatter(
            dim=1,
            index=top_k_indices,
            src=combined_weights,
        )

        moe_output = torch.sum(
            expert_outputs * router_weights.unsqueeze(-1),
            dim=1,
        )

        return moe_output, clean_router_logits, top_k_indices


class MoEModel(nn.Module):
    """
    ONNX-friendly MoE classifier.

    The constructor and forward signature match src/models/moe/model.py so this
    class can be used as a drop-in replacement for inference/export.
    """

    def __init__(
        self,
        context_dim: Optional[int],
        num_classes: int,
        num_experts: int,
        top_k: int,
        router_mode: Literal["noisy", "context_aware"],
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_mode = router_mode
        self.temperature = temperature

        self.feature_extractor = Mobilenetv3SmallFeatureExtractor(
            pretrained=True,
            freeze_backbone=False,
        )
        model_dim = self.feature_extractor.output_dim

        self.pre_moe_norm = nn.LayerNorm(model_dim)
        self.post_moe_norm = nn.LayerNorm(model_dim)

        self.moe_layer = MoELayer(
            context_dim=context_dim,
            model_dim=model_dim,
            num_experts=num_experts,
            top_k=top_k,
            router_mode=router_mode,
            temperature=temperature,
        )

        self.classifier = nn.Sequential(
            nn.Linear(model_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature = self.feature_extractor(x)
        residual = feature

        feature_norm = self.pre_moe_norm(feature)

        if self.router_mode == "noisy":
            moe_output, clean_router_logits, top_k_indices = self.moe_layer(
                feature_norm,
            )
        elif self.router_mode == "context_aware":
            moe_output, clean_router_logits, top_k_indices = self.moe_layer(
                feature_norm,
                context,
            )
        else:
            raise ValueError(
                f"Invalid router_mode: {self.router_mode}. "
                "Must be 'noisy' or 'context_aware'."
            )

        moe_residual = residual + moe_output
        moe_residual_norm = self.post_moe_norm(moe_residual)
        class_logits = self.classifier(moe_residual_norm)

        return class_logits, clean_router_logits, top_k_indices

