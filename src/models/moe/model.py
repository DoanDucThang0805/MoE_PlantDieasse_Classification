"""Mixture of Experts (MoE) model for plant disease classification.

This module implements a flexible MoE architecture with gating networks
for routing input data to specialized expert networks in parallel.
"""

import warnings
from typing import Tuple

import torch
import torch.nn as nn
from torchinfo import summary

from .backbone import Mobilenetv3SmallFeatureExtractor
from .gating import NoisyTopKGating

warnings.filterwarnings("ignore")

# ============================================================================
# CONSTANTS
# ============================================================================
DEFAULT_EMBEDDING_DIM = 576        # MobileNetV3 Small feature dimension
DEFAULT_EXPERT_HIDDEN_DIM = 576*2   # Hidden layer multiplier (2x embedding_dim)
DEFAULT_CLASSIFIER_HIDDEN_DIM = 120  # Classifier hidden dimension
DEFAULT_DROPOUT_RATE = 0.1
DEFAULT_NUM_CLASSES = 8
DEFAULT_NUM_EXPERTS = 2
DEFAULT_TOP_K = 2
DEFAULT_BATCH_SIZE = 32
DEFAULT_IMAGE_SIZE = 224
DEFAULT_NUM_IMAGE_CHANNELS = 3


class MoELayer(nn.Module):
    """Mixture of Experts layer with noisy top-k gating.
    
    Routes input features to the top-k expert networks based on learned
    gating weights. Each expert is a small feed-forward network.
    
    Args:
        embedding_dim (int): Feature dimension for routing and processing.
        num_experts (int): Number of expert networks.
        top_k (int): Number of top experts to use for each input sample.
    """
    
    def __init__(self, embedding_dim: int, num_experts: int, top_k: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature

        # Gating network for expert selection
        self.gating = NoisyTopKGating(
            model_dim=embedding_dim,
            num_experts=num_experts,
            top_k=top_k,
            temperature=temperature
        )
        
        # Expert networks - each is a feed-forward network
        self.experts = self._build_experts(embedding_dim, num_experts)
    
    @staticmethod
    def _build_experts(embedding_dim: int, num_experts: int) -> nn.ModuleList:
        """Build expert feed-forward networks.
        
        Args:
            embedding_dim (int): Input/output dimension.
            num_experts (int): Number of experts to create.
            
        Returns:
            nn.ModuleList: List of expert networks.
        """
        return nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim * 2),
                nn.ReLU(),
                nn.Linear(embedding_dim * 2, embedding_dim)
            )
            for _ in range(num_experts)
        ])


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through MoE layer.
        
        Args:
            x (torch.Tensor): Shape [batch_size, embedding_dim].
            
        Returns:
            Tuple of:
                - moe_output (torch.Tensor): [batch_size, embedding_dim]
                - clean_logits (torch.Tensor): [batch_size, num_experts]
                - top_k_indices (torch.Tensor): [batch_size, top_k]
        """
        # Get gating decisions and routing weights
        combined_weights, top_k_indices, clean_logits = self.gating(x)
        
        # Initialize output accumulator
        moe_output = torch.zeros_like(x)
        
        # Route through experts
        for expert_idx in range(self.num_experts):
            # Create mask for samples routed to this expert
            expert_mask = (top_k_indices == expert_idx)  # [batch_size, top_k]
            sample_mask = expert_mask.any(dim=1)         # [batch_size]
            
            # Skip if no samples are routed to this expert
            if not sample_mask.any():
                continue
            
            # Process selected samples through expert
            selected_features = x[sample_mask]
            expert_output = self.experts[expert_idx](selected_features)
            
            # Compute expert weights and aggregate contributions
            expert_weights = (combined_weights * expert_mask).sum(dim=1)
            selected_weights = expert_weights[sample_mask]
            
            # Accumulate weighted expert outputs
            moe_output[sample_mask] += expert_output * selected_weights.unsqueeze(-1)
        
        return moe_output, clean_logits, top_k_indices


class MoEModel(nn.Module):
    """Mixture of Experts model for plant disease classification.
    
    Architecture:
        1. Feature Extractor: MobileNetV3 Small backbone
        2. Pre-normalization: Layer normalization
        3. MoE Layer: Routes features through multiple expert networks
        4. Residual connection: Skip connection from backbone
        5. Post-normalization: Layer normalization
        6. Classifier: MLP head for classification
    
    Args:
        num_classes (int): Number of output classes.
        num_experts (int, optional): Number of expert networks. Default: 4.
        top_k (int, optional): Number of top experts to use. Default: 3.
    """
    
    def __init__(
        self,
        num_classes: int,
        num_experts: int,
        top_k: int,
        temperature: float = 1.0
    ) -> None:
        super().__init__()
        
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature
        
        # Feature extraction backbone
        self.feature_extractor = Mobilenetv3SmallFeatureExtractor(
            pretrained=True,
            freeze_backbone=False
        )
        embedding_dim = self.feature_extractor.output_dim
        
        # Normalization layers
        self.pre_normalizer = nn.LayerNorm(embedding_dim)
        self.post_normalizer = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(DEFAULT_DROPOUT_RATE)
        
        # MoE routing layer
        self.moe_layer = MoELayer(
            embedding_dim=embedding_dim,
            num_experts=num_experts,
            top_k=top_k,
            temperature=temperature
        )
        
        # Classification head
        self.classifier = self._build_classifier(embedding_dim, num_classes)
    
    @staticmethod
    def _build_classifier(embedding_dim: int, num_classes: int) -> nn.Sequential:
        """Build classifier MLP.
        
        Args:
            embedding_dim (int): Input dimension.
            num_classes (int): Output dimension.
            
        Returns:
            nn.Sequential: Classifier network.
        """
        hidden_dim = 32
        return nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the MoE model.
        
        Pipeline:
            Image → Feature Extraction → Pre-norm → MoE → Residual →
            Post-norm → Classifier → Logits
        
        Args:
            x (torch.Tensor): Input images [batch_size, 3, 224, 224].
            
        Returns:
            Tuple of:
                - logits (torch.Tensor): [batch_size, num_classes]
                - clean_logits (torch.Tensor): [batch_size, num_experts]
                - top_k_indices (torch.Tensor): [batch_size, top_k]
        """
        # Extract features from image backbone
        features = self.feature_extractor(x)  # [batch_size, embedding_dim]
        residual = features
        
        # Apply pre-normalization and MoE routing
        normalized = self.pre_normalizer(features)
        moe_output, clean_logits, top_k_indices = self.moe_layer(normalized)
        
        # Apply dropout and residual connection
        moe_output = self.dropout(moe_output)
        moe_output = moe_output + residual
        
        # Apply post-normalization and classification
        moe_output = self.post_normalizer(moe_output)
        logits = self.classifier(moe_output)
        
        return logits, clean_logits, top_k_indices


def test_moe_layer() -> None:
    """Test MoE layer functionality."""
    print("\n" + "=" * 70)
    print("TEST 1: Mixture of Experts Layer")
    print("=" * 70)
    
    # Create and test MoE layer
    batch_size = 3
    moe_layer = MoELayer(
        embedding_dim=DEFAULT_EMBEDDING_DIM,
        num_experts=DEFAULT_NUM_EXPERTS,
        top_k=DEFAULT_TOP_K
    )
    
    dummy_input = torch.randn(batch_size, DEFAULT_EMBEDDING_DIM)
    moe_output, clean_logits, top_k_indices = moe_layer(dummy_input)
    
    print(f"Input shape:           {dummy_input.shape}")
    print(f"MoE Output shape:      {moe_output.shape}")
    print(f"Gating Logits shape:   {clean_logits.shape}")
    print(f"Top-k Indices shape:   {top_k_indices.shape}")
    print("✓ MoE Layer test passed!\n")


def test_moe_model() -> None:
    """Test full MoE model pipeline."""
    print("=" * 70)
    print("TEST 2: Full MoE Model Pipeline")
    print("=" * 70)
    
    # Create and test full model
    model = MoEModel(
        num_classes=DEFAULT_NUM_CLASSES,
        num_experts=DEFAULT_NUM_EXPERTS,
        top_k=DEFAULT_TOP_K
    )
    
    dummy_images = torch.randn(
        DEFAULT_BATCH_SIZE,
        DEFAULT_NUM_IMAGE_CHANNELS,
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE
    )
    
    class_logits, gating_logits, top_k_indices = model(dummy_images)
    
    print(f"Input shape:           {dummy_images.shape}")
    print(f"Class Logits shape:    {class_logits.shape}")
    print(f"Gating Logits shape:   {gating_logits.shape}")
    print(f"Top-k Indices shape:   {top_k_indices.shape}")
    print("\n✓ Full MoE Model pipeline test passed!")
    
    # Print model summary
    print("\n" + "=" * 70)
    print("Model Architecture Summary")
    print("=" * 70)
    summary(
        model,
        input_size=(1, DEFAULT_NUM_IMAGE_CHANNELS, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
        col_names=["input_size", "output_size", "num_params", "trainable"]
    )
    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_moe_layer()
    test_moe_model()
