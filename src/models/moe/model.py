"""
Mixture of Experts (MoE) model for plant disease classification.

This module implements a MoE architecture that routes input features through
multiple specialized experts using a learned gating mechanism.
"""

import torch
import torch.nn as nn
from typing import Tuple, Literal, Optional, Union
from .backbone import Mobilenetv3LargeFeatureExtractor, Mobilenetv3SmallFeatureExtractor, EfficientNetV2MFeatureExtractor
from .gating import NoisyTopKGating, ContextAwareGating
import warnings

warnings.filterwarnings("ignore")


class MoELayer(nn.Module):
    """
    Mixture of Experts (MoE) layer with expert routing and selective combination.

    This layer enables the model to use multiple specialist experts and a gating
    network to intelligently route data to the most relevant experts for each input.
    
    The architecture includes:
    - A gating network that learns to route inputs to appropriate experts
    - Multiple feed-forward expert networks that specialize on different features
    - A combination mechanism to merge expert outputs based on gating weights
    
    Args:
        context_dim (int): Dimension of context features for context-aware gating.
        model_dim (int): Hidden feature dimension from the backbone network.
        num_experts (int): Number of expert networks in the mixture.
        top_k (int): Number of top experts to select for each input sample.
        router_mode (Literal["noisy", "context_aware"]): Routing strategy to use.
            - "noisy": Uses noisy top-k gating with auxiliary loss.
            - "context_aware": Uses context information for adaptive routing.
    """
    
    def __init__(
        self, 
        context_dim: Union[int, None],
        model_dim: int, 
        num_experts: int, 
        top_k: int,
        router_mode: Literal["noisy", "context_aware"],
        temperature: float = 1.0
    ) -> None:
        
        """Initialize the MoE layer with specified configuration."""
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_mode = router_mode
        self.temperature = temperature

        if not (0 < self.top_k <= self.num_experts):
            raise ValueError(
                "top_k must be a positive integer less than or equal to num_experts"
            )

        # Initialize gating network based on routing strategy
        self._initialize_gating(model_dim, context_dim)

        # Create list of expert networks - each is a feed-forward network
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(model_dim, 1024),  # Expand to 2x dimension
                nn.LayerNorm(1024),
                nn.GELU(),                   # Non-linear activation
                nn.Dropout(0.1),             # Regularization
                nn.Linear(1024, model_dim)   # Contract back to original dimension
            ) 
            for _ in range(num_experts)
        ])

    def _initialize_gating(self, model_dim: int, context_dim: Optional[int]) -> None:
        """Initialize the appropriate gating network based on router mode."""
        if self.router_mode == "noisy":
            self.gating = NoisyTopKGating(
                model_dim=model_dim,
                num_experts=self.num_experts, 
                top_k=self.top_k
            )
        elif self.router_mode == "context_aware":
            self.gating = ContextAwareGating(
                model_dim=model_dim,
                context_dim=context_dim,
                num_experts=self.num_experts, 
                top_k=self.top_k,
                temperature=self.temperature
            )
        else:
            raise ValueError(
                f"Invalid router_mode: {self.router_mode}. "
                "Must be 'noisy' or 'context_aware'."
            )

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through MoE layer with intelligent router-based expert selection.

        Performs the following steps:
        1. Route inputs through gating network to get expert selection weights
        2. Initialize output tensor for accumulating expert contributions
        3. For each expert: select relevant samples, process through expert network,
           scale by routing weights, and accumulate in output
        4. Apply residual connection to preserve input information
        
        Args:
            x (torch.Tensor): Input features of shape [batch_size, model_dim].
            context (Optional[torch.Tensor]): Context features of shape [batch_size, context_dim].
                Required when router_mode is "context_aware". Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - moe_output (torch.Tensor): Expert-processed features [batch_size, model_dim].
                - clean_router_logits (torch.Tensor): Gating network logits [batch_size, num_experts].
                - top_k_indices (torch.Tensor): Indices of selected experts [batch_size, top_k].
        """
        # Step 1: Obtain routing decisions from gating network
        if self.router_mode == "noisy":
            combined_weights, top_k_indices, clean_router_logits = self.gating(x)
        elif self.router_mode == "context_aware":
            combined_weights, top_k_indices, clean_router_logits = self.gating(x, context)
        else:
            raise ValueError(
                f"Invalid router_mode: {self.router_mode}. "
                "Must be 'noisy' or 'context_aware'."
            )

        # Step 2: Initialize output tensor for accumulating expert contributions
        moe_output = torch.zeros_like(x)
        
        # Step 3: Route inputs through selected experts and combine outputs
        for expert_idx in range(self.num_experts):
            # Create mask to identify samples routed to this expert
            expert_mask = (top_k_indices == expert_idx)      # [batch_size, top_k]
            sample_mask = expert_mask.any(dim=1)             # [batch_size] - which samples are routed
            
            # Skip if no samples are routed to this expert (optimization)
            if sample_mask.sum() == 0:
                continue
            
            # Extract features for samples routed to this expert
            selected_features = x[sample_mask]
            
            # Process through expert feed-forward network
            expert_output = self.experts[expert_idx](selected_features)
            
            # Calculate expert weights from gating network
            expert_weights = (combined_weights * expert_mask).sum(dim=1)
            selected_weights = expert_weights[sample_mask]
            
            # Accumulate weighted expert contribution to final output
            moe_output[sample_mask] += expert_output * selected_weights.unsqueeze(-1)
        
        return moe_output, clean_router_logits, top_k_indices


class MoEModel(nn.Module):
    """
    Mixture of Experts model for plant disease leaf classification.

    This model combines a backbone feature extractor (MobileNetV3 Large) with a
    Mixture of Experts layer to enable adaptive, specialized feature processing.
    
    Architecture pipeline:
    1. Feature Extractor: MobileNetV3 Large to extract semantic features from images
    2. Pre-MoE Normalization: Layer normalization to stabilize inputs to MoE layer
    3. MoE Layer: Route features through multiple experts based on gating mechanism
    4. Residual Connection: Add expert-processed features back to input features
    5. Post-MoE Normalization: Layer normalization before classification head
    6. Classifier: Linear layer to produce class predictions
    
    Args:
        context_dim (int): Dimension of context features for adaptive routing.
        num_classes (int): Number of plant disease classes to predict.
        num_experts (int): Number of expert networks in the MoE layer.
        top_k (int): Number of top experts to select for each sample.
        router_mode (Literal["noisy", "context_aware"]): Type of routing mechanism.
            Defaults to "noisy". Use "context_aware" for context-dependent routing.
    """
    
    def __init__(
        self, 
        context_dim: Union[int, None],
        num_classes: int, 
        num_experts: int, 
        top_k: int, 
        router_mode: Literal["noisy", "context_aware"],
        temperature: float = 1.0
    ) -> None:
        
        """Initialize MoE model with specified configuration."""
        super().__init__()
        self.context_dim = context_dim
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_mode = router_mode
        self.temperature = temperature

        # Create feature extractor from MobileNetV3 Small backbone
        self.feature_extractor = Mobilenetv3SmallFeatureExtractor(
            pretrained=True, 
            freeze_backbone=False
        )
        model_dim = self.feature_extractor.output_dim

        # Normalization layers to stabilize feature distributions
        self.pre_moe_norm = nn.LayerNorm(model_dim)   # Before MoE routing
        self.post_moe_norm = nn.LayerNorm(model_dim)  # After expert processing

        # Initialize MoE layer based on selected routing strategy
        self.moe_layer = MoELayer(
            context_dim=context_dim,
            model_dim=model_dim, 
            num_experts=num_experts, 
            top_k=top_k, 
            router_mode=router_mode,
            temperature=temperature
        )
        
        # Classification head: maps features to class logits
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )


    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the complete Mixture of Experts classification pipeline.

        Processing pipeline:
        Image Input → Feature Extraction → Pre-MoE Norm → MoE Layer → Residual Addition →
        Post-MoE Norm → Classification → Logits

        The model extracts semantic features from images, routes them through experts
        based on learned gating, and combines expert outputs for final classification.
        
        Args:
            x (torch.Tensor): Input images of shape [batch_size, 3, 224, 224].
            context (Optional[torch.Tensor]): Context features of shape [batch_size, context_dim].
                Required when router_mode is "context_aware". Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - logits (torch.Tensor): Classification logits [batch_size, num_classes].
                - clean_logits (torch.Tensor): Gating network logits [batch_size, num_experts].
                - top_k_indices (torch.Tensor): Selected expert indices [batch_size, top_k].
        """
        # Step 1: Extract semantic features from input images
        # Input:  [batch_size, 3, 224, 224]
        # Output: [batch_size, model_dim] (typically 960 for MobileNetV3 Large)
        feature = self.feature_extractor(x)
        residual = feature  # Save for residual connection
        
        # Step 2: Normalize features to stabilize MoE layer inputs
        feature_norm = self.pre_moe_norm(feature)
        
        # Step 3: Route features through expert networks based on gating mechanism
        # This step adaptively selects and combines expert outputs for feature enhancement
        if self.router_mode == "noisy":
            moe_output, clean_router_logits, top_k_indices = self.moe_layer(feature_norm)
        elif self.router_mode == "context_aware":
            moe_output, clean_router_logits, top_k_indices = self.moe_layer(feature_norm, context)
        else:
            raise ValueError(
                f"Invalid router_mode: {self.router_mode}. "
                "Must be 'noisy' or 'context_aware'."
            )
        
        # Step 4: Apply residual connection to preserve original feature information
        # while combining with expert-enhanced features
        moe_residual = residual + moe_output

        # Step 5: Normalize expert-combined features before classification
        moe_residual_norm = self.post_moe_norm(moe_residual)
        
        # Step 6: Classify normalized features into disease categories
        # Input:  [batch_size, model_dim]
        # Output: [batch_size, num_classes]
        class_logits = self.classifier(moe_residual_norm)
        return class_logits, clean_router_logits, top_k_indices
    