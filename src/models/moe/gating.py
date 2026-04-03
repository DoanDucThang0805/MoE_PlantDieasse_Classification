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
import numpy as np


class StandardTopKgating(nn.Module):
    """
    Standard top-k gating mechanism for Mixture of Experts (MoE).
    
    This gating module implements a simple and deterministic expert selection mechanism.
    It uses a learned linear projection to compute logits for all experts, then selects
    and normalizes the k experts with the highest logits.
    
    Key Characteristics:
    - Deterministic: No randomness in expert selection
    - Reproducible: Same input always produces same output
    - Efficient: Single linear layer computation
    - Baseline: Suitable as a reference implementation
    
    Use Cases:
    - Baseline comparisons against other gating mechanisms
    - Inference-only scenarios requiring deterministic routing
    - Scenarios where exploration (noise) is not needed/desired
    
    Design:
    - Input is projected to expert logits via a learned linear layer
    - Top-k selection ensures load balancing by selecting exactly k experts
    - Softmax normalization produces valid probability weights
    """
    
    def __init__(self, model_dim: int, num_experts: int, top_k: int) -> None:
        """
        Initialize the standard top-k gating module.
        
        Constructs a linear layer that transforms input features to expert logits,
        without bias terms for computational efficiency.
        
        Args:
            model_dim (int): Dimension of input feature vectors (e.g., 960 for ResNet backbone)
            num_experts (int): Total number of available experts (e.g., 4 or 8)
            top_k (int): Number of top experts to select (e.g., 2 or 3)
        
        Raises:
            AssertionError: If top_k > num_experts (cannot select more experts than available)
        """
        super().__init__()
        assert top_k <= num_experts, "top_k must be less than or equal to num_experts"
        
        self.model_dim = model_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Linear projection layer that transforms input features to expert scores
        # bias=False reduces parameters and typically improves stability
        self.gate_projector = nn.Linear(self.model_dim, self.num_experts, bias=False)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route input to selected experts using top-k gating.
        
        Processing Pipeline:
        1. **Project features to expert logits**: Linear transformation produces raw scores for each expert
        2. **Select top-k experts**: Identifies the k experts with highest scores
        3. **Normalize via softmax**: Converts selected expert logits to probability weights (sum=1)
        
        The output weights can be used to create a weighted sum of expert outputs for the final prediction.
        
        Args:
            x (torch.Tensor): Input feature tensor
                            Shape: [batch_size, model_dim]
                            Contains the fused/aggregated features from the backbone network
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                
                (1) combined_weights: Softmax-normalized probability weights for selected experts
                    - Shape: [batch_size, top_k]
                    - Values: In range [0, 1], sum to 1 along dimension 1
                    - Interpretation: How much weight each selected expert receives
                
                (2) top_k_indices: Integer indices of the selected experts
                    - Shape: [batch_size, top_k]
                    - Values: In range [0, num_experts-1]
                    - Use to index into the expert network outputs
                
                (3) gate_logits: Raw pre-softmax logits for all experts (for auxiliary losses)
                    - Shape: [batch_size, num_experts]
                    - Values: Unbounded real numbers (can be negative or very large)
                    - Useful for computing load balancing losses
        
        Example:
            >>> batch_size, model_dim, num_experts, top_k = 32, 960, 8, 3
            >>> gating = StandardTopKgating(model_dim, num_experts, top_k)
            >>> x = torch.randn(batch_size, model_dim)  # Feature vectors
            >>> weights, indices, logits = gating(x)
            >>> print(weights.shape)  # torch.Size([32, 3])
            >>> print(indices.shape)  # torch.Size([32, 3])
            >>> print(logits.shape)   # torch.Size([32, 8])
        """
        # Compute logits (raw expert scores) for all experts via linear projection
        gate_logits = self.gate_projector(x)
        
        # Select the top-k experts based on highest logits
        # torch.topk returns (values, indices) where values are sorted in descending order
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        
        # Normalize top-k logits to probability weights using softmax
        # Ensures weights sum to 1 and are interpretable as probabilities
        combined_weights = F.softmax(top_k_logits, dim=-1, dtype=torch.float32)
        
        return combined_weights, top_k_indices, gate_logits
    

class NoisyTopKGating(nn.Module):
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
    
    def __init__(self, model_dim: int, num_experts: int, top_k: int, noise_stddev=1.0) -> None:
        """
        Initialize the noisy top-k gating module.
        
        Sets up two linear projector layers:
        1. gate_projector: Produces the deterministic (clean) expert scores
        2. noise_layer: Produces per-expert noise scaling factors
        
        Args:
            model_dim (int): Dimension of input feature vectors (e.g., 960)
            num_experts (int): Total number of available experts (e.g., 8)
            top_k (int): Number of top experts to select (e.g., 3)
            noise_stddev (float, optional): Multiplier for the noise magnitude.
                                          Controls how much noise is added relative to learned scale.
                                          Higher values = more exploration. Defaults to 1.0.
        
        Raises:
            AssertionError: If top_k > num_experts
        
        Note:
            The noise_stddev is a hyperparameter that should be tuned. Typical values:
            - 0.5-1.0: Mild exploration (recommended for stable training)
            - 1.0-2.0: Moderate exploration (balances exploitation and exploration)
            - 2.0+: Aggressive exploration (may hurt performance if too high)
        """
        super().__init__()
        assert top_k <= num_experts, "top_k must be less than or equal to num_experts"
        
        self.model_dim = model_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_stddev = noise_stddev

        # Linear layer: computes deterministic expert selection logits
        # This is the primary routing signal
        self.gate_projector = nn.Linear(model_dim, num_experts, bias=False)
        
        # Linear layer: learns to predict noise magnitude for each expert
        # These magnitudes are adaptively learned and can vary per expert
        self.noise_layer = nn.Linear(model_dim, num_experts, bias=False)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route input to selected experts using noisy top-k gating.
        
        This method branches into two different behaviors based on the training mode:
        
        **Training Mode (self.training = True):**
        - Adds learnable Gaussian noise to encourage exploration
        - Noise magnitude is predicted per-expert and adaptively learned
        - Different samples may select different experts even if similar
        - Helps prevent load imbalance and over-specialization
        
        **Inference Mode (self.training = False):**
        - Uses deterministic clean logits (no noise)
        - Identical inputs always produce identical routing decisions
        - Fast and reproducible inference
        
        Processing Pipeline:
        
        1. **Compute Clean Logits**: Linear projection produces base expert scores
        
        2. **Conditional Noise Addition** (training only):
           - Predict noise magnitude via learned linear layer
           - Apply softplus(·) to ensure positive scaling: softplus(x) = log(1 + e^x)
           - Sample i.i.d. Gaussian noise N(0, 1)
           - Scale noise by learned magnitudes and stddev multiplier
           - Add to clean logits: logits_noisy = logits_clean + scale × noise × stddev
        
        3. **Top-k Selection**: Select k experts with highest (possibly noisy) logits
        
        4. **Softmax Normalization**: Convert selected logits to probability weights
        
        Args:
            x (torch.Tensor): Input feature tensor
                            Shape: [batch_size, model_dim]
                            Batched feature vectors from backbone network
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                
                (1) combined_weights: Normalized probability weights for selected experts
                    - Shape: [batch_size, top_k]
                    - Values summed to 1 per sample via softmax
                    - Interpretation: How much weight each selected expert receives
                
                (2) top_k_indices: Indices of the selected experts
                    - Shape: [batch_size, top_k]
                    - Integer indices in range [0, num_experts-1]
                
                (3) clean_logits: Raw (noise-free) expert scores
                    - Shape: [batch_size, num_experts]
                    - Useful for computing auxiliary losses (load balancing)
                    - Independent of training mode (always clean)
        
        Training vs Inference Difference:
            Training:  same input → different outputs (due to noise) → exploration
            Inference: same input → identical outputs (no noise) → consistency
        
        Example:
            >>> gating = NoisyTopKGating(model_dim=960, num_experts=8, top_k=3, noise_stddev=1.0)
            >>> x = torch.randn(32, 960)
            >>> gating.train()   # Training mode
            >>> w1, idx1, _ = gating(x)
            >>> w2, idx2, _ = gating(x)
            >>> # w1 and w2 will likely be different due to noise
            >>> gating.eval()    # Inference mode  
            >>> w3, idx3, _ = gating(x)
            >>> w4, idx4, _ = gating(x)
            >>> # w3 and w4 will be identical (no noise)
        """

        # Compute base expert logits via learned linear transformation
        # These represent the deterministic preference for each expert
        clean_logits = self.gate_projector(x)

        # Only add noise during training phase
        if self.training:
            # Predict per-expert noise scaling factors based on input
            noise_magnitude = self.noise_layer(x)
            
            # Apply softplus to ensure noise scale is strictly positive
            # softplus(x) = log(1 + exp(x)) is smooth and always positive
            # This prevents negative scaling which would invert the noise effect
            noise_scale = torch.nn.functional.softplus(noise_magnitude)

            # Sample independent Gaussian noise for all experts and samples
            # Shape matches clean_logits: [batch_size, num_experts]
            sampled_noise = torch.randn_like(clean_logits)

            # Combine clean logits with scaled noise for exploration
            # Each expert gets its own learned noise magnitude
            noisy_logits = clean_logits + noise_scale * sampled_noise * self.noise_stddev
        else:
            # During inference, use clean logits without any noise
            # Ensures deterministic and reproducible predictions
            noisy_logits = clean_logits

        # Select the top-k logits and corresponding expert indices
        # torch.topk automatically handles ties consistently
        top_k_logits, top_k_indices = torch.topk(noisy_logits, self.top_k, dim=-1)

        # Convert selected logits to normalized probability weights via softmax
        # Ensures weights are in [0, 1] and sum to 1 per sample
        combined_weights = F.softmax(top_k_logits, dim=-1)

        # Return expert weights, their indices, and clean logits for auxiliary losses
        return combined_weights, top_k_indices, clean_logits    


class ContextAwareGating(nn.Module):
    """
    Context-aware gating mechanism for Mixture of Experts.
    
    This gating module enhances expert selection by incorporating contextual information
    alongside the main embedding. It fuses embedding and context features to make more
    informed decisions about which experts to route to.
    
    Architecture:
    - Normalizes embedding and context separately using LayerNorm
    - Projects context features to a fixed dimension (32)
    - Concatenates embedding and projected context
    - Applies multi-layer MLP to predict expert logits
    - Optionally adds Gaussian noise during training for exploration
    - Selects top-k experts with highest logits
    
    Attributes:
        model_dim (int): Dimension of input embeddings (e.g., 960)
        context_dim (int): Dimension of context features
        num_experts (int): Total number of experts available
        top_k (int): Number of top experts to select
        noise_stddev (float): Standard deviation of noise added during training
    """
    
    def __init__(self, model_dim: int, context_dim: int, num_experts: int, top_k: int, noise_stddev=1.0) -> None:
        """
        Initialize the context-aware gating module.
        
        Args:
            model_dim (int): Dimension of input embeddings
            context_dim (int): Dimension of context feature vectors
            num_experts (int): Total number of experts in the MoE layer
            top_k (int): Number of top experts to select based on logits
            noise_stddev (float, optional): Standard deviation of Gaussian noise added during training.
                                           Defaults to 1.0.
        
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
        fusion_dim = model_dim + 32

        # Layer normalization for input embedding
        self.embedding_norm = nn.LayerNorm(model_dim)
        # Layer normalization for context features
        self.context_norm = nn.LayerNorm(context_dim)
        # Layer normalization for fused embedding and context
        self.fusion_norm = nn.LayerNorm(fusion_dim)

        # Project context features to 32 dimensions
        # Uses GELU activation for smoothness
        self.context_projector = nn.Sequential(
            nn.Linear(context_dim, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU()
        )

        # Predict noise magnitude for each expert
        # Enables context-aware noise in gating decisions
        self.noise_layer = nn.Linear(fusion_dim, num_experts, bias=False)
        
        # Multi-layer MLP to predict logits for all experts
        # Takes fused representation and outputs expert logits
        self.gate_projector = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim//2),
            nn.GELU(),
            nn.Linear(fusion_dim//2, fusion_dim//4),
            nn.GELU(),
            nn.Linear(fusion_dim//4, num_experts)
        )

    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate gating weights for expert selection using context information.
        
        Processing pipeline:
        1. Normalize input embedding and context independently
        2. Project context features to 32 dimensions
        3. Concatenate (embedding + projected_context)
        4. Normalize the fused representation
        5. Generate clean logits via multi-layer MLP
        6. During training: Add learnable Gaussian noise to logits for exploration
        7. Select top-k experts and compute softmax weights
        
        Args:
            x (torch.Tensor): Input embedding tensor of shape [batch_size, model_dim]
            context (torch.Tensor): Context feature tensor of shape [batch_size, context_dim]
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - combined_weights: Softmax normalized weights for top-k experts 
                                  Shape: [batch_size, top_k]
                - top_k_indices: Indices of selected top-k experts 
                                Shape: [batch_size, top_k]
                - clean_logits: Raw logits for all experts (before noise) 
                               Shape: [batch_size, num_experts]
        """
        
        embedding = self.embedding_norm(x)
        context = self.context_norm(context)

        # Project context to fixed dimension
        context_features = self.context_projector(context)
        # Concatenate normalized embedding with projected context
        fusion_features = torch.cat([embedding, context_features], dim=-1)
        # Normalize the fused representation
        fusion_features = self.fusion_norm(fusion_features)

        # Compute clean logits (without noise)
        clean_logits = self.gate_projector(fusion_features)
        
        # Add noise during training for improved exploration
        if self.training:
            # Predict noise magnitude based on fused features
            noise_magnitude = self.noise_layer(fusion_features)
            # Ensure positive scaling using softplus: log(1 + exp(x))
            noise_scale = nn.functional.softplus(noise_magnitude)
            # Sample Gaussian noise
            sampled_noise = torch.randn_like(clean_logits)
            # Combine clean logits with scaled noise
            noisy_logits = clean_logits + noise_scale * sampled_noise * self.noise_stddev
        else:
            # Use clean logits without noise during inference
            noisy_logits = clean_logits
        
        # Select top-k experts
        top_k_logits, top_k_indices = torch.topk(noisy_logits, self.top_k, dim=-1)
        # Normalize top-k logits to weights via softmax
        combined_weights = F.softmax(top_k_logits, dim=-1)

        return combined_weights, top_k_indices, clean_logits
