"""
MoE Model Inference and Evaluation Module
========================================

This module performs inference on a test dataset using a trained MoE model
and generates comprehensive evaluation reports including classification metrics,
confusion matrices, and performance visualizations.

Features:
    - Load trained model checkpoints
    - Batch inference on test data
    - Generate classification reports
    - Visualize confusion matrices
    - Create performance heatmaps
    - Export detailed results to reports directory
"""

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from dataset.plantdoc_dataset import test_dataset
from models.moe.model import MoEModel


# ============================================================================
# Logging Configuration
# ============================================================================

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# ============================================================================
# Constants
# ============================================================================

# Model and checkpoint information
MODEL_NAME = 'mobilenetv3large_moe'
RUN_TIME = 'run_20260320-155951'  # Timestamp of training run
DATASET_NAME = 'plantdoc'

# Data loading parameters
BATCH_SIZE = 32
SHUFFLE_TEST = True

# Visualization parameters
CONFUSION_MATRIX_FIGSIZE = (12, 10)
CLASSIFICATION_REPORT_FIGSIZE = (10, 6)
REPORT_DPI = 300
PLOT_FORMAT = 'png'

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# ============================================================================
# CLI and Path Configuration
# ============================================================================

def get_args() -> Any:
    parser = ArgumentParser(
        description="Run MoE inference for a saved checkpoint and generate evaluation reports."
    )
    parser.add_argument(
        "--model-name",
        "--modelname",
        dest="model_name",
        type=str,
        default=MODEL_NAME,
        help="Model directory name under checkpoints/MoE and reports/MoE."
    )
    parser.add_argument(
        '--type-model',
        dest="type_model",
        type=str,
        default="MoE",
        help='Loại mô hình để huấn luyện (ví dụ: MoE, pretrauined, v.v.)'
    )
    parser.add_argument(
        "--run-time",
        "--runtime",
        dest="run_time",
        type=str,
        default=RUN_TIME,
        help="Training run timestamp folder name."
    )
    parser.add_argument(
        "--dataset-name",
        "--datasetname",
        dest="dataset_name",
        type=str,
        default=DATASET_NAME,
        help="Dataset name used under checkpoints and reports directories."
    )
    parser.add_argument(
        "--topk",
        dest="top_k",
        type=int,
        default=None,
        help="Number of top experts to load for the MoE model."
    )
    parser.add_argument(
        "--numexpert",
        "--num-expert",
        dest="num_experts",
        type=int,
        default=None,
        help="Number of experts used by the MoE model."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


def get_checkpoint_path(
    dataset_name: str,
    model_name: str,
    type_model: str,
    run_time: str,
    num_experts: int = None,
    top_k: int = None,
    seed: int = 42
) -> Path:
    """
    Get the checkpoint file path based on configuration.
    
    Args:
        dataset_name: Dataset name under checkpoints.
        model_name: Model folder name under MoE.
        run_time: Training run timestamp folder name.
        num_experts: Number of experts (optional).
        top_k: Number of top experts (optional).
        seed: Random seed (optional).
    Returns:
        Path to the checkpoint file
    """
    checkpoint_base = (
        Path(__file__).resolve().parents[3] / 'checkpoints' / dataset_name / type_model / model_name
    )

    if num_experts is not None and top_k is not None:
        return checkpoint_base / f"{num_experts}_experts" / f"top_{top_k}" / f"seed_{seed}" / run_time / 'best_checkpoint.pth'

    if not checkpoint_base.exists():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {checkpoint_base}"
        )

    for expert_dir in sorted(checkpoint_base.iterdir()):
        if not expert_dir.is_dir() or not expert_dir.name.endswith("_experts"):
            continue
        if num_experts is not None and expert_dir.name != f"{num_experts}_experts":
            continue
        for top_dir in sorted(expert_dir.iterdir()):
            if not top_dir.is_dir() or not top_dir.name.startswith("top_"):
                continue
            if top_k is not None and top_dir.name != f"top_{top_k}":
                continue
            candidate = top_dir / run_time / 'best_checkpoint.pth'
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        f"No matching checkpoint found in {checkpoint_base} for num_experts={num_experts}, top_k={top_k}, run_time={run_time}"
    )


def get_report_dir(
    dataset_name: str,
    model_name: str,
    type_model: str,
    run_time: str,
    num_experts: int,
    top_k: int,
    seed: int
) -> Path:
    """Get the report directory path based on model configuration."""
    return (
        Path(__file__).resolve().parents[3] / 'reports' / dataset_name / type_model / 
        model_name / f"{num_experts}_experts" / f"top_{top_k}" / f"seed_{seed}" / run_time
    )


# ============================================================================
# Helper Functions
# ============================================================================

def load_checkpoint(checkpoint_path: Path) -> Tuple[Dict[str, Any], int, int, int, float]:
    """
    Load model checkpoint and extract model configuration.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Tuple of (state_dict, num_classes, num_experts, top_k, temperature)
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        KeyError: If required keys are missing in checkpoint
    """
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Support different state_dict key formats
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    top_k = checkpoint.get("top_k")
    num_experts = checkpoint.get("num_experts")
    num_classes = checkpoint.get("num_classes")
    temperature = checkpoint.get("temperature")
    logger.info(
        f"Checkpoint loaded: num_classes={num_classes}, "
        f"num_experts={num_experts}, top_k={top_k}, temperature={temperature}"
    )
    
    return state_dict, num_classes, num_experts, top_k, temperature


def initialize_model(
    state_dict: Dict[str, Any],
    num_classes: int,
    num_experts: int,
    top_k: int,
    temperature: float
) -> torch.nn.Module:
    """
    Initialize and load the MoE model.
    
    Args:
        state_dict: Model state dictionary
        num_classes: Number of output classes
        num_experts: Number of expert networks
        top_k: Number of top experts to select
        temperature: Temperature for gating softmax
    Returns:
        Loaded model on the configured device in eval mode
    """
    model = MoEModel(
        num_classes=num_classes,
        num_experts=num_experts,
        top_k=top_k,
        temperature=temperature
    )
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    
    logger.info("Model initialized and loaded successfully")
    return model


def create_test_dataloader() -> DataLoader:
    """
    Create a dataloader for the test dataset.
    
    Returns:
        DataLoader configured for test inference
    """
    logger.info(f"Creating test dataloader with batch_size={BATCH_SIZE}")
    return DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE_TEST
    )


def perform_inference(
    model: torch.nn.Module,
    test_loader: DataLoader
) -> Tuple[List, List]:
    """
    Perform inference on test dataset.
    
    Args:
        model: Trained MoE model
        test_loader: DataLoader with test samples
        
    Returns:
        Tuple of (all_predictions, all_labels) as lists
    """
    all_preds = []
    all_labels = []
    
    logger.info("Starting inference on test dataset...")
    
    with torch.inference_mode(True):
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Model returns: logits, auxiliary_loss, expert_assignment
            logits, _, _ = model(images)
            
            # Compute predictions from logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {batch_idx + 1} batches")
    
    logger.info(f"Inference complete. Total samples processed: {len(all_labels)}")
    return all_preds, all_labels


def save_plot(
    filepath: Path,
    filename: str,
    description: str = "plot"
) -> None:
    """
    Save and display current matplotlib figure.
    
    Args:
        filepath: Directory where file will be saved
        filename: Name of the output file (without extension)
        description: Human-readable description of the plot
    """
    output_path = filepath / f"{filename}.{PLOT_FORMAT}"
    plt.savefig(output_path, dpi=REPORT_DPI, bbox_inches="tight")
    logger.info(f"Saved {description}: {output_path}")
    plt.show()


def visualize_confusion_matrix(
    labels: List,
    predictions: List,
    target_names: List[str],
    report_dir: Path
) -> None:
    """
    Create and save confusion matrix visualization.
    
    Args:
        labels: True labels
        predictions: Predicted labels
        target_names: Names of target classes
        report_dir: Directory to save the visualization
    """
    logger.info("Generating confusion matrix...")
    
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=CONFUSION_MATRIX_FIGSIZE)
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
    
    save_plot(report_dir, "confusion_matrix", "confusion matrix")


def visualize_classification_report(
    labels: List,
    predictions: List,
    target_names: List[str],
    report_dir: Path
) -> None:
    """
    Create and save classification report heatmap.
    
    Args:
        labels: True labels
        predictions: Predicted labels
        target_names: Names of target classes
        report_dir: Directory to save the visualization
    """
    logger.info("Generating classification report heatmap...")
    
    report_dict = classification_report(
        labels,
        predictions,
        target_names=target_names,
        output_dict=True
    )
    
    df = pd.DataFrame(report_dict).transpose()
    
    plt.figure(figsize=CLASSIFICATION_REPORT_FIGSIZE)
    sns.heatmap(
        df.iloc[:-1, :-1],
        annot=True,
        cmap="Blues",
        fmt=".2f",
        cbar_kws={"label": "Value"}
    )
    
    plt.title("Classification Report (Precision / Recall / F1-score)", fontsize=12)
    plt.xlabel("Evaluation Metrics")
    plt.ylabel("Disease Classes")
    plt.tight_layout()
    
    save_plot(report_dir, "classification_report_heatmap", "classification report")


# ============================================================================
# Main Execution
# ============================================================================

def main() -> None:
    """
    Main inference pipeline: load model, perform inference, generate reports.
    """
    logger.info("=" * 80)
    logger.info("Starting MoE Model Inference and Evaluation")
    logger.info("=" * 80)
    
    try:
        args = get_args()
        logger.info(
            f"Inference configuration: dataset_name={args.dataset_name}, model_name={args.model_name}, type_model={args.type_model}, "
            f"run_time={args.run_time}, num_experts={args.num_experts}, top_k={args.top_k}, seed={args.seed}"
        )

        # Load checkpoint and extract configuration
        checkpoint_path = get_checkpoint_path(
            dataset_name=args.dataset_name,
            model_name=args.model_name,
            type_model=args.type_model,
            run_time=args.run_time,
            num_experts=args.num_experts,
            top_k=args.top_k,
            seed=args.seed
        )
        state_dict, num_classes, num_experts, top_k, temperature = load_checkpoint(checkpoint_path)
        
        # Initialize model
        model = initialize_model(state_dict, num_classes, num_experts, top_k, temperature)
        
        # Load test data
        test_loader = create_test_dataloader()
        
        # Perform inference
        all_preds, all_labels = perform_inference(model, test_loader)
        
        # Get class names
        target_names = [test_dataset.idx_to_class[i] for i in range(len(test_dataset.idx_to_class))]
        logger.info(f"Total classes: {len(target_names)}")
        
        # Create and setup report directory
        report_dir = get_report_dir(
            dataset_name=args.dataset_name,
            model_name=args.model_name,
            type_model=args.type_model,
            run_time=args.run_time,
            num_experts=num_experts,
            top_k=top_k,
            seed=args.seed
        )
        report_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Report directory: {report_dir}")
        
        # Print classification report to console
        logger.info("\n" + "=" * 80)
        logger.info("CLASSIFICATION REPORT")
        logger.info("=" * 80)
        report_text = classification_report(all_labels, all_preds, target_names=target_names)
        logger.info("\n" + report_text)
        
        # Generate visualizations
        visualize_confusion_matrix(all_labels, all_preds, target_names, report_dir)
        visualize_classification_report(all_labels, all_preds, target_names, report_dir)
        
        logger.info("=" * 80)
        logger.info("Model Evaluation Complete")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
