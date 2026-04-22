"""
Gating Mechanisms for Mixture of Experts (MoE).

This module provides three gating mechanisms for routing inputs to experts in a MoE layer:

1. StandardTopKgating: Simple top-k selection based on learned logits
   - Deterministic expert selection during inference
   - Selects exactly k experts with highest scores

2. NoisyTopKGating: Top-k selection with noise injection during training
   - Adds Gaussian noise during training for exploration
   - Learned noise magnitude per expert
   - Deterministic during inference (no noise)

3. ContextAwareGating: Gating that incorporates contextual information
   - Fuses embedding with context features
   - More informed expert selection
   - Supports both training and inference modes

All gating mechanisms follow the same interface:
    Input: feature tensor
    Output: (expert_weights, expert_indices, logits)

Author: MoE Team
Date: 2026-03-31
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseNoiseGatingMixin:
    """Mixin to handle noise injection logic for gating mechanisms."""
    
    @staticmethod
    def apply_noise_to_logits(
        clean_logits: torch.Tensor,
        noise_layer: nn.Module,
        noise_stddev: float,
        training: bool,
        input_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply learnable Gaussian noise to logits during training.
        
        Args:
            clean_logits: Raw expert logits without noise [batch_size, num_experts]
            noise_layer: Linear layer that predicts noise magnitude
            noise_stddev: Multiplier for noise strength
            training: Whether in training mode
            input_features: Input features to predict noise magnitude
            
        Returns:
            Noisy logits during training, clean logits during inference
        """
        if not training:
            return clean_logits
        
        # Predict per-expert noise scaling factors
        noise_magnitude = noise_layer(input_features)
        # Apply softplus to ensure strictly positive scaling
        noise_scale = torch.nn.functional.softplus(noise_magnitude)
        # Sample independent Gaussian noise
        sampled_noise = torch.randn_like(clean_logits)
        # Combine with scaled noise
        noisy_logits = clean_logits + noise_scale * sampled_noise * noise_stddev
        
        return noisy_logits


class NoisyTopKGating(nn.Module, BaseNoiseGatingMixin):
    """
    Top-k gating mechanism with learnable noise injection for Mixture of Experts.
    
    This gating module enhances exploration during training by adding learnable Gaussian noise
    to the expert selection logits. Noise is only applied during training; inference is
    deterministic and noise-free.
    
    Architecture:
    - gate_projector: Learns clean expert logits without noise
    - noise_layer: Learns the scale of noise to add per expert
    - During training: Combined logits = clean logits + (learned noise scale) × (Gaussian noise)
    - During inference: Uses clean logits only
    
    Key Benefits:
    1. **Improved Exploration**: Noise encourages trying different expert combinations
    2. **Load Balancing**: Prevents the same experts from always being selected
    3. **Expert Specialization**: Different experts develop distinct expertise
    4. **Generalization**: Stochastic training → better generalization at test time
    5. **Deterministic Inference**: No randomness once deployed
    
    Comparison to Standard Gating:
    - Standard: Always selects the exact same k experts for identical inputs
    - Noisy: During training selects different experts with some probability
    - Result: More robust and well-distributed expert usage
    """
    
    def __init__(self, model_dim: int, num_experts: int, top_k: int, temperature: float = 1.0, noise_stddev=1.0, hidden_dim: int = 128, intermediate_dim: int = 32) -> None:
        """
        Initialize the noisy top-k gating module.
        
        Args:
            model_dim (int): Dimension of input feature vectors (e.g., 960)
            num_experts (int): Total number of available experts (e.g., 8)
            top_k (int): Number of top experts to select (e.g., 3)
            temperature (float, optional): Temperature for softmax. Defaults to 1.0.
            noise_stddev (float, optional): Multiplier for noise magnitude. Defaults to 1.0.
            hidden_dim (int, optional): First hidden layer dimension. Defaults to 128.
            intermediate_dim (int, optional): Intermediate layer dimension. Defaults to 32.
        
        Raises:
            AssertionError: If top_k > num_experts
        """
        super().__init__()
        assert top_k <= num_experts, "top_k must be less than or equal to num_experts"
        
        self.model_dim = model_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_stddev = noise_stddev
        self.temperature = temperature

        # Multi-layer MLP for expert selection logits
        self.gate_projector = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, num_experts)
        )
        
        # Learns per-expert noise magnitude
        self.noise_layer = nn.Linear(model_dim, num_experts, bias=False)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route input to selected experts using noisy top-k gating.
        
        Args:
            x: Input feature tensor of shape [batch_size, model_dim]
        
        Returns:
            Tuple of:
                - combined_weights: Softmax normalized weights [batch_size, top_k]
                - top_k_indices: Indices of selected experts [batch_size, top_k]
                - clean_logits: Raw expert logits [batch_size, num_experts]
        """
        # Compute clean expert logits
        clean_logits = self.gate_projector(x)
        
        # Apply noise during training for exploration, deterministic during inference
        noisy_logits = self.apply_noise_to_logits(
            clean_logits, self.noise_layer, self.noise_stddev, self.training, x
        )
        
        # Select top-k experts
        top_k_logits, top_k_indices = torch.topk(noisy_logits, self.top_k, dim=-1)
        
        # Normalize to probabilities
        combined_weights = F.softmax(top_k_logits / self.temperature, dim=-1)
        
        return combined_weights, top_k_indices, clean_logits    


class ContextAwareGating(nn.Module, BaseNoiseGatingMixin):
    """
    Context-aware gating mechanism for Mixture of Experts.
    
    Enhances expert selection by incorporating contextual information alongside
    the main embedding. Fuses both signals to make informed routing decisions.
    """
    
    def __init__(
        self,
        model_dim: int,
        context_dim: int,
        num_experts: int,
        top_k: int,
        temperature: float = 1.0,
        noise_stddev: float = 1.0,
        hidden_dim: int = 128,
        intermediate_dim: int = 32,
        context_proj_dim: int = 32
    ) -> None:
        """
        Initialize the context-aware gating module.
        
        Args:
            model_dim: Dimension of input embeddings
            context_dim: Dimension of context feature vectors
            num_experts: Total number of experts in MoE layer
            top_k: Number of top experts to select
            temperature: Temperature for softmax normalization
            noise_stddev: Standard deviation of noise during training
            hidden_dim: First hidden layer dimension
            intermediate_dim: Intermediate layer dimension
            context_proj_dim: Dimension for context projection
        
        Raises:
            AssertionError: If top_k > num_experts
        """
        super().__init__()
        assert top_k <= num_experts, "top_k must be less than or equal to num_experts"

        self.model_dim = model_dim
        self.context_dim = context_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_stddev = noise_stddev
        self.temperature = temperature
        
        fusion_dim = model_dim + context_proj_dim

        # Normalize inputs independently
        self.embedding_norm = nn.LayerNorm(model_dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.context_proj_norm = nn.LayerNorm(context_proj_dim)
        self.fusion_norm = nn.LayerNorm(fusion_dim)

        # Project context to fixed dimension
        self.context_projector = nn.Sequential(
            nn.Linear(context_dim, context_proj_dim),
            nn.GELU(),
            nn.Linear(context_proj_dim, context_proj_dim),
        )

        # Predict noise magnitude from fused features
        self.noise_layer = nn.Linear(fusion_dim, num_experts, bias=False)
        
        # Multi-layer MLP for expert selection
        self.gate_projector = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, num_experts)
        )

    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate gating weights for expert selection using context information.
        
        Args:
            x: Input embedding tensor of shape [batch_size, model_dim]
            context: Context feature tensor of shape [batch_size, context_dim]
        
        Returns:
            Tuple of:
                - combined_weights: Softmax normalized weights [batch_size, top_k]
                - top_k_indices: Indices of selected experts [batch_size, top_k]
                - clean_logits: Raw expert logits [batch_size, num_experts]
        """
        # Normalize inputs
        embedding = self.embedding_norm(x)
        context = self.context_norm(context)

        # Project and normalize context
        context_features = self.context_projector(context)
        context_features = self.context_proj_norm(context_features)

        # Fuse embedding and context
        fusion_features = torch.cat([embedding, context_features], dim=-1)
        fusion_features = self.fusion_norm(fusion_features)
        
        # Compute clean logits
        clean_logits = self.gate_projector(fusion_features)
        
        # Apply noise during training
        noisy_logits = self.apply_noise_to_logits(
            clean_logits, self.noise_layer, self.noise_stddev, self.training, fusion_features
        )
        
        # Select top-k experts
        top_k_logits, top_k_indices = torch.topk(noisy_logits, self.top_k, dim=-1)
        # Normalize with temperature scaling
        combined_weights = F.softmax(top_k_logits / self.temperature, dim=-1)

        return combined_weights, top_k_indices, clean_logits
