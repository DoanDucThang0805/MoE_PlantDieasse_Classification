"""
MoE Model Evaluation and Inference Module
==========================================

This module performs inference and comprehensive evaluation of a trained Mixture of Experts (MoE)
model on a test dataset. It generates evaluation reports including classification metrics,
confusion matrices, and performance visualizations.

Features:
    - Load pretrained MoE model checkpoint
    - Batch inference on test data
    - Generate detailed classification reports
    - Visualize confusion matrices
    - Create performance metrics heatmaps
    - Handle multiple checkpoint formats
    - Context-aware and non-context inference modes

Usage:
    python context_aware_moe_inference.py --use_context True --router_mode context_aware

Author: MoE Team
Version: 2.0
"""

import logging
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import argparse

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from dataset.plantdoc_dataset import build_datasets
from models.moe.model import MoEModel

# Configure logging for production use
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Management
# ============================================================================

class Config:
    """
    Production configuration management for model inference.
    
    This class centralizes all configuration parameters for inference,
    including model specifications, data loading parameters, and output settings.
    
    Attributes:
        model_name (str): Name of the model architecture (default: 'mobilenetv3large_moe')
        run_time (str): Timestamp identifier of the training run (default: 'run_20260320-155951')
        dataset_name (str): Name of the dataset (default: 'plantdoc')
        num_classes (int): Number of classification classes (default: 8)
        num_experts (int): Number of experts in MoE model (default: 8)
        top_k (int): Number of experts selected per input (default: 4)
        batch_size (int): Batch size for inference (default: 32)
        shuffle_test (bool): Whether to shuffle test data (default: True)
        confusion_matrix_figsize (tuple): Figure size for confusion matrix plot (default: (12, 10))
        classification_report_figsize (tuple): Figure size for classification report plot (default: (10, 6))
        report_dpi (int): DPI for saved report images (default: 300)
        device (str): Computing device ('cuda' or 'cpu')
        
    Methods:
        get_checkpoint_path(): Get the path to model checkpoint
        get_report_dir(): Get the directory for saving reports
    """
    
    # Model configuration
    model_name: str = 'mobilenetv3large_moe'
    type_model: str = 'MoE'
    run_time: str = 'run_20260402-135005'
    dataset_name: str = 'plantdoc'
    
    # Model architecture parameters
    num_classes: int = 8
    num_experts: int = 6
    context_dim: int = 6
    top_k: int = 2
    
    # Data loading parameters
    batch_size: int = 32
    shuffle_test: bool = True
    
    # Visualization parameters
    confusion_matrix_figsize: Tuple[int, int] = (12, 10)
    classification_report_figsize: Tuple[int, int] = (10, 6)
    report_dpi: int = 300
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    @classmethod
    def get_checkpoint_path(cls) -> Path:
        """
        Get the path to the model checkpoint.
        
        Returns:
            Path: Full path to the best checkpoint file
            
        Raises:
            FileNotFoundError: If checkpoint does not exist
        """
        checkpoint_path = (
            Path(__file__).resolve().parents[3] / 'checkpoints' / cls.dataset_name / cls.type_model / 
            cls.model_name / cls.run_time / 'best_checkpoint.pth'
        )
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at: {checkpoint_path}\n"
                f"Please verify model_name, run_time, and dataset_name configurations."
            )
        
        return checkpoint_path
    
    @classmethod
    def get_report_dir(cls) -> Path:
        """
        Get the directory for saving evaluation reports.
        
        Creates the directory if it doesn't exist.
        
        Returns:
            Path: Directory path for saving reports
        """
        report_dir = (
            Path(__file__).resolve().parents[3] / 'reports' / cls.dataset_name / cls.type_model / 
            cls.model_name / cls.run_time
        )
        report_dir.mkdir(parents=True, exist_ok=True)
        return report_dir


# ============================================================================
# Argument Parser
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for inference configuration.
    
    Returns:
        argparse.Namespace: Parsed arguments containing:
            - use_context (bool): Whether to use context features
            - router_mode (str): Router mode for MoE gating
            
    Raises:
        argparse.ArgumentTypeError: If invalid argument values provided
    """
    parser = argparse.ArgumentParser(
        description="Evaluate trained MoE model on test dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--use_context",
        type=bool,
        default=True,
        choices=[True, False],
        help="Whether to use context features (default: True)"
    )
    parser.add_argument(
        "--router_mode",
        type=str,
        default="context_aware",
        choices=["context_aware", "noisy"],
        help="Router mode for MoE gating (default: context_aware)"
    )
    
    return parser.parse_args()


# ============================================================================
# Data Loading Functions
# ============================================================================

def setup_test_dataloader(use_context: bool) -> Tuple[DataLoader, object]:
    """
    Setup test data loader and dataset.
    
    Args:
        use_context (bool): Whether to load context features
        
    Returns:
        Tuple[DataLoader, object]: Test DataLoader and test dataset object
        
    Raises:
        RuntimeError: If dataset loading fails
    """
    try:
        logger.info("Loading datasets...")
        _, _, test_dataset = build_datasets(use_context=use_context)
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.batch_size,
            shuffle=Config.shuffle_test,
            num_workers=0,
            pin_memory=True if 'cuda' in Config.device else False
        )
        
        logger.info(f"Dataset loaded successfully. Test set size: {len(test_dataset)}")
        return test_loader, test_dataset
        
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        raise RuntimeError(f"Dataset loading failed: {e}") from e


def get_class_names(test_dataset: object) -> List[str]:
    """
    Extract class names from the test dataset.
    
    Args:
        test_dataset (object): Test dataset object with class mapping
        
    Returns:
        List[str]: List of class names in order
        
    Raises:
        AttributeError: If dataset doesn't have idx_to_class mapping
    """
    try:
        class_names = [
            test_dataset.idx_to_class[i] 
            for i in range(len(test_dataset.idx_to_class))
        ]
        logger.info(f"Found {len(class_names)} disease classes")
        return class_names
    except (AttributeError, KeyError) as e:
        logger.error(f"Failed to extract class names: {e}")
        raise


# ============================================================================
# Model Loading Functions
# ============================================================================

def create_model(num_experts: int, top_k: int, router_mode: str) -> MoEModel:
    """
    Create and initialize MoE model architecture.
    
    Args:
        num_experts (int): Number of experts in MoE model
        top_k (int): Number of experts selected per input
        router_mode (str): Router mode ('context_aware' or 'noisy')
    
    Returns:
        MoEModel: Uninitialized MoE model instance
        
    Raises:
        RuntimeError: If model creation fails
    """
    try:
        logger.info("Creating MoE model architecture...")
        logger.info(f"  Num Experts: {num_experts}, Top-K: {top_k}, Router Mode: {router_mode}")
        
        model = MoEModel(
            context_dim=Config.context_dim,
            num_classes=Config.num_classes,
            num_experts=num_experts,
            top_k=top_k,
            router_mode=router_mode
        )
        logger.info(f"Model created successfully on device: {Config.device}")
        return model
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        raise RuntimeError(f"Model creation failed: {e}") from e


def extract_checkpoint_metadata(checkpoint_path: Path) -> Dict[str, any]:
    """
    Extract model hyperparameters from checkpoint metadata.
    
    Loads checkpoint and extracts num_experts, top_k, and router_mode
    with fallback to Config defaults if not available.
    
    Args:
        checkpoint_path (Path): Path to checkpoint file
        
    Returns:
        Dict: Dictionary containing 'num_experts', 'top_k', 'router_mode'
        
    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        RuntimeError: If checkpoint loading fails
    """
    try:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint metadata from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=Config.device)
        
        # Extract metadata with fallbacks
        metadata = {
            'num_experts': Config.num_experts,
            'top_k': Config.top_k,
            'router_mode': 'context_aware'
        }
        
        # Check for metadata in checkpoint
        if isinstance(checkpoint, dict):
            if 'num_experts' in checkpoint:
                metadata['num_experts'] = checkpoint['num_experts']
                logger.info(f"Loaded num_experts from checkpoint: {metadata['num_experts']}")
            
            if 'top_k' in checkpoint:
                metadata['top_k'] = checkpoint['top_k']
                logger.info(f"Loaded top_k from checkpoint: {metadata['top_k']}")
            
            if 'router_mode' in checkpoint:
                metadata['router_mode'] = checkpoint['router_mode']
                logger.info(f"Loaded router_mode from checkpoint: {metadata['router_mode']}")
        
        logger.info(f"Using model configuration: {metadata}")
        return metadata
        
    except FileNotFoundError as e:
        logger.error(f"Checkpoint file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to extract checkpoint metadata: {e}")
        raise RuntimeError(f"Checkpoint metadata extraction failed: {e}") from e


def load_checkpoint(model: MoEModel, checkpoint_path: Path) -> MoEModel:
    """
    Load model weights from checkpoint.
    
    Supports multiple checkpoint formats and provides detailed error handling.
    
    Args:
        model (MoEModel): Model to load weights into
        checkpoint_path (Path): Path to checkpoint file
        
    Returns:
        MoEModel: Model with loaded weights, moved to configured device in eval mode
        
    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        RuntimeError: If checkpoint format is invalid
        torch.cuda.OutOfMemoryError: If device memory insufficient
    """
    try:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading model weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=Config.device)
        
        # Handle multiple checkpoint formats for model weights
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                logger.info("Loaded 'model_state_dict' from checkpoint")
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                logger.info("Loaded 'state_dict' from checkpoint")
            else:
                # Check if it's a full checkpoint with metadata but no model_state_dict key
                # In this case, filter out non-state_dict keys
                state_dict = {k: v for k, v in checkpoint.items() 
                             if not isinstance(v, (str, int, bool, dict)) or k == "state_dict"}
                if not state_dict:
                    state_dict = checkpoint
                logger.info("Loaded checkpoint directly as state_dict")
        else:
            state_dict = checkpoint
        
        # Load state dict into model
        model.load_state_dict(state_dict)
        model = model.to(Config.device)
        model.eval()
        
        logger.info("Model weights loaded successfully and set to evaluation mode")
        return model
        
    except FileNotFoundError as e:
        logger.error(f"Checkpoint file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise RuntimeError(f"Checkpoint loading failed: {e}") from e


# ============================================================================
# Inference Function
# ============================================================================

def run_inference(
    model: MoEModel,
    test_loader: DataLoader,
    use_context: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform inference on test dataset.
    
    Runs model inference in no-gradient mode for efficiency and produces
    predictions on all test samples.
    
    Args:
        model (MoEModel): Model in evaluation mode
        test_loader (DataLoader): Test data loader
        use_context (bool): Whether context features are used in dataset
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of predictions and true labels
        
    Raises:
        RuntimeError: If inference fails
    """
    all_preds = []
    all_labels = []
    
    try:
        logger.info("Running inference on test dataset...")
        
        with torch.inference_mode():
            for batch_idx, batch in enumerate(test_loader):
                # Handle batch structure
                if use_context:
                    images, labels, context = batch
                    images = images.to(Config.device)
                    labels = labels.to(Config.device)
                    context = context.to(Config.device) if context is not None else None
                    
                    # Forward pass with context
                    logits, _, _ = model(images, context)
                else:
                    images, labels = batch
                    images = images.to(Config.device)
                    labels = labels.to(Config.device)
                    
                    # Forward pass without context
                    logits, _, _ = model(images)
                
                # Convert logits to predictions
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                # Store results
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                
                if (batch_idx + 1) % 10 == 0:
                    logger.debug(f"Processed {(batch_idx + 1) * Config.batch_size} samples")
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        logger.info(
            f"Inference completed. Processed {len(all_labels)} samples. "
            f"Accuracy: {(all_preds == all_labels).mean():.4f}"
        )
        
        return all_preds, all_labels
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise RuntimeError(f"Inference failed: {e}") from e


# ============================================================================
# Reporting Functions
# ============================================================================

def generate_classification_report(
    all_labels: np.ndarray,
    all_preds: np.ndarray,
    target_names: List[str]
) -> Dict:
    """
    Generate detailed classification report.
    
    Args:
        all_labels (np.ndarray): True labels
        all_preds (np.ndarray): Predicted labels
        target_names (List[str]): Class names
        
    Returns:
        Dict: Classification report as dictionary
    """
    logger.info("Generating classification report...")
    
    report_text = classification_report(
        all_labels,
        all_preds,
        target_names=target_names
    )
    
    print("\n" + "=" * 80)
    print("CLASSIFICATION REPORT")
    print("=" * 80)
    print(report_text)
    
    return classification_report(
        all_labels,
        all_preds,
        target_names=target_names,
        output_dict=True
    )


def save_confusion_matrix(
    all_labels: np.ndarray,
    all_preds: np.ndarray,
    target_names: List[str],
    report_dir: Path
) -> None:
    """
    Generate and save confusion matrix visualization.
    
    Args:
        all_labels (np.ndarray): True labels
        all_preds (np.ndarray): Predicted labels
        target_names (List[str]): Class names
        report_dir (Path): Directory to save the figure
        
    Raises:
        RuntimeError: If visualization fails
    """
    try:
        logger.info("Generating confusion matrix visualization...")
        
        cm = confusion_matrix(all_labels, all_preds)
        
        plt.figure(figsize=Config.confusion_matrix_figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names
        )
        
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.title("Confusion Matrix - Plant Disease Classification", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        output_path = report_dir / "confusion_matrix.png"
        plt.savefig(output_path, dpi=Config.report_dpi, bbox_inches="tight")
        logger.info(f"✓ Confusion matrix saved: {output_path}")
        plt.close()
        
    except Exception as e:
        logger.error(f"Failed to save confusion matrix: {e}")
        raise RuntimeError(f"Confusion matrix visualization failed: {e}") from e


def save_classification_report_heatmap(
    report_dict: Dict,
    target_names: List[str],
    report_dir: Path
) -> None:
    """
    Generate and save classification report heatmap.
    
    Args:
        report_dict (Dict): Classification report as dictionary
        target_names (List[str]): Class names
        report_dir (Path): Directory to save the figure
        
    Raises:
        RuntimeError: If visualization fails
    """
    try:
        logger.info("Generating classification report heatmap...")
        
        df = pd.DataFrame(report_dict).transpose()
        
        plt.figure(figsize=Config.classification_report_figsize)
        sns.heatmap(
            df.iloc[:-1, :-1],
            annot=True,
            cmap="Blues",
            fmt=".2f",
            cbar_kws={"label": "Score"}
        )
        
        plt.title("Classification Report (Precision / Recall / F1-score)", fontsize=12)
        plt.xlabel("Evaluation Metric")
        plt.ylabel("Disease Class")
        plt.tight_layout()
        
        output_path = report_dir / "classification_report_heatmap.png"
        plt.savefig(output_path, dpi=Config.report_dpi, bbox_inches="tight")
        logger.info(f"✓ Classification report saved: {output_path}")
        plt.close()
        
    except Exception as e:
        logger.error(f"Failed to save classification report heatmap: {e}")
        raise RuntimeError(f"Classification report visualization failed: {e}") from e


# ============================================================================
# Main Execution Function
# ============================================================================

def main():
    """
    Main execution function orchestrating the entire inference pipeline.
    
    Pipeline:
        1. Parse command-line arguments
        2. Setup data loader and dataset
        3. Create and load model
        4. Run inference
        5. Generate and save reports
    
    Raises:
        RuntimeError: If any step in the pipeline fails
    """
    try:
        logger.info("=" * 80)
        logger.info("Starting MoE Model Evaluation")
        logger.info("=" * 80)
        logger.info(f"Configuration:")
        logger.info(f"  Model: {Config.model_name}")
        logger.info(f"  Run: {Config.run_time}")
        logger.info(f"  Device: {Config.device}")
        logger.info(f"  Num Classes: {Config.num_classes}")
        
        # Parse arguments
        args = parse_arguments()
        logger.info(f"  Use Context: {args.use_context}")
        logger.info(f"  Router Mode: {args.router_mode}")
        
        # Setup data
        test_loader, test_dataset = setup_test_dataloader(args.use_context)
        target_names = get_class_names(test_dataset)
        
        # Load checkpoint metadata
        checkpoint_path = Config.get_checkpoint_path()
        metadata = extract_checkpoint_metadata(checkpoint_path)
        
        # Create model with parameters from checkpoint
        model = create_model(
            num_experts=metadata['num_experts'],
            top_k=metadata['top_k'],
            router_mode=metadata['router_mode']
        )
        
        # Load model weights
        model = load_checkpoint(model, checkpoint_path)
        
        # Run inference
        all_preds, all_labels = run_inference(model, test_loader, args.use_context)
        
        # Generate reports
        report_dict = generate_classification_report(all_labels, all_preds, target_names)
        
        # Save visualizations
        report_dir = Config.get_report_dir()
        save_confusion_matrix(all_labels, all_preds, target_names, report_dir)
        save_classification_report_heatmap(report_dict, target_names, report_dir)
        
        logger.info("=" * 80)
        logger.info("Model Evaluation Completed Successfully")
        logger.info(f"Reports saved to: {report_dir}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.critical(f"Fatal error during evaluation: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
