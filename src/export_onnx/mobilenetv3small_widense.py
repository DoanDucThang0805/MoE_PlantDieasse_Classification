"""
Module for exporting MobileNetV3-Small with Widened Dense Classifier Head to ONNX format.

This module provides functionality to export a trained MobileNetV3-Small model with 
a widened multi-layer dense classifier head to ONNX (Open Neural Network Exchange) format, 
enabling deployment on various inference engines and platforms.

The model architecture consists of:
- MobileNetV3-Small backbone (pre-trained)
- Custom widened dense classifier head:
  * Linear(576 -> 2048) + Hardswish + Dropout(0.2)
  * Linear(2048 -> 512) + Hardswish + Dropout(0.2)  
  * Linear(512 -> 8) [8 plant disease classes]

Author: Context MoE Plant Disease Classification Team
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
from models.pretrained_model.widense_mlp_head import model


def loadcheckpoint(model: nn.Module, checkpoint_path: Path) -> nn.Module:
    """
    Load model checkpoint from file and apply weights to the model.
    
    Args:
        model (nn.Module): The model architecture to load weights into.
        checkpoint_path (Path): Path to the checkpoint file containing model state dict.
    
    Returns:
        nn.Module: The model with loaded weights applied.
    
    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        RuntimeError: If checkpoint format is incompatible with model.
    """
    # Determine device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint from file to appropriate device
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Apply saved weights to model
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return model


def export_mobilenetv3small_widense_to_onnx(
    output_path: Path,
    checkpoint_path: Path | None = None,
    num_classes: int = 8,
    opset_version: int = 18,
) -> None:
    """
    Export MobileNetV3-Small with widened dense classifier head to ONNX format.
    
    This function takes a MobileNetV3-Small model with custom widened dense classifier
    head and exports it to ONNX format, which can be used for inference on various 
    platforms and devices. The exported model can be used with ONNX Runtime for 
    efficient inference.
    
    Model Architecture:
    - Input: RGB image tensor (batch_size, 3, 224, 224)
    - Backbone: MobileNetV3-Small (pre-trained on ImageNet)
    - Classifier Head (widened dense):
      * Linear(576 -> 2048) with Hardswish activation and Dropout
      * Linear(2048 -> 512) with Hardswish activation and Dropout
      * Linear(512 -> 8) output layer [8 plant disease classes]
    - Output: Class probabilities (batch_size, 8)
    
    Args:
        output_path (Path): Path where the ONNX model file will be saved.
        checkpoint_path (Path | None): Optional path to checkpoint file to load pre-trained weights.
                                       If None, uses model with random initialization.
        num_classes (int): Number of output classes for the classification task. Default: 8 (plant diseases).
        opset_version (int): ONNX opset version to use. Default: 18 (compatible with recent ONNX Runtime).
    
    Returns:
        None
    
    Raises:
        IOError: If output directory cannot be created or is not writable.
        RuntimeError: If ONNX export fails during model tracing.
    
    Example:
        >>> export_mobilenetv3small_widense_to_onnx(
        ...     output_path=Path("./onnx/mobilenetv3small_widense.onnx"),
        ...     checkpoint_path=Path("./checkpoints/model.pt"),
        ...     num_classes=8,
        ...     opset_version=18
        ... )
    """
    # Initialize model instance
    model_instance = model
    
    # Load pre-trained weights if checkpoint path provided
    if checkpoint_path is not None:
        model_instance = loadcheckpoint(model_instance, checkpoint_path)
        print(f"✓ Loaded checkpoint: {checkpoint_path}")
    
    # Set model to evaluation mode (disables dropout, batch norm updates)
    model_instance.eval()
    
    # Create output directory if it doesn't exist
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine device for model inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_instance = model_instance.to(device)
    
    # Create dummy input tensor for model tracing
    # Input shape: (batch_size=1, channels=3, height=224, width=224) for MobileNetV3
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    
    # Export model to ONNX format
    try:
        torch.onnx.export(
            model_instance,
            dummy_input,
            output_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=opset_version,
            do_constant_folding=True,  # Optimize by folding constant expressions
            verbose=False,
            dynamic_axes={  # Allow variable batch size
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )
        print(f"✓ Successfully exported model to ONNX format: {output_path}")
        print(f"  - Model: MobileNetV3-Small with Widened Dense Classifier Head")
        print(f"  - Classifier: Linear(576->2048)->ReLU->Linear(2048->512)->ReLU->Linear(512->{num_classes})")
        print(f"  - Number of classes: {num_classes}")
        print(f"  - ONNX opset version: {opset_version}")
        print(f"  - Input shape: (batch_size, 3, 224, 224)")
        print(f"  - Output shape: (batch_size, {num_classes})")
        
    except Exception as e:
        print(f"✗ Error during ONNX export: {str(e)}")
        raise


if __name__ == "__main__":
    """
    Example usage: Export MobileNetV3-Small with widened dense head model with optional checkpoint loading.
    """
    # Define paths
    output_onnx_path = Path("./onnx/mobilenetv3small_widense.onnx")
    checkpoint_path = Path("./checkpoints/plantdoc/pretrain_weight/model.pt")  # Optional
    
    print("=" * 80)
    print("MobileNetV3-Small with Widened Dense Classifier Head - ONNX Export")
    print("=" * 80)
    
    # Export model
    export_mobilenetv3small_widense_to_onnx(
        output_path=output_onnx_path,
        checkpoint_path=checkpoint_path if checkpoint_path.exists() else None,
        num_classes=8,
        opset_version=18,
    )
    
    print("=" * 80)

