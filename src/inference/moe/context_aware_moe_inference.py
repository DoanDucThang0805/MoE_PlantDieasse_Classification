"""
MoE Model Inference and Evaluation

Performs inference and comprehensive evaluation of trained MoE models on test datasets.
Generates classification reports, confusion matrices, and performance visualizations.
"""

import logging
from pathlib import Path
from typing import Tuple, Dict, List
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Inference configuration management."""
    
    # Model parameters
    model_name: str = 'mobilenetv3large_moe'
    type_model: str = 'MoE'
    run_time: str = 'run_20260402-170406'
    dataset_name: str = 'plantdoc'
    seed: int = 42
    
    # Model hyperparameters (from checkpoint)
    num_classes: int = 8
    num_experts: int = 8
    context_dim: int = 6
    top_k: int = 2
    temperature: float = 1.0
    router_mode: str = 'context_aware'
    use_context: bool = True
    
    # Data & visualization
    batch_size: int = 32
    shuffle_test: bool = True
    confusion_matrix_figsize: Tuple[int, int] = (12, 10)
    classification_report_figsize: Tuple[int, int] = (10, 6)
    report_dpi: int = 300
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def update_from_args(cls, args: argparse.Namespace) -> None:
        """Update configuration from CLI arguments."""
        for key, value in vars(args).items():
            if hasattr(cls, key) and value is not None:
                setattr(cls, key, value)
        
        logger.info(f"Config: model={cls.model_name}, experts={cls.num_experts}, "
                   f"top_k={cls.top_k}, device={cls.device}")

    @classmethod
    def get_checkpoint_path(cls) -> Path:
        """Get checkpoint path."""
        path = (
            Path(__file__).resolve().parents[3] / 'checkpoints' / cls.dataset_name / 
            cls.type_model / cls.model_name / f'{cls.num_experts}_experts' / 
            f'top_{cls.top_k}' / f'seed_{cls.seed}' / cls.run_time / 'best_checkpoint.pth'
        )
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    @classmethod
    def get_report_dir(cls) -> Path:
        """Get or create report directory."""
        path = (
            Path(__file__).resolve().parents[3] / 'reports' / cls.dataset_name / 
            cls.type_model / cls.model_name / f'{cls.num_experts}_experts' / 
            f'top_{cls.top_k}' / f'seed_{cls.seed}' / cls.run_time
        )
        path.mkdir(parents=True, exist_ok=True)
        return path


# ============================================================================
# Argument Parser
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for inference."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained MoE model on test dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python context_aware_moe_inference.py --run_time run_20260320-155951
  python context_aware_moe_inference.py --model_name mobilenetv3small_moe --run_time run_20260317-224514
        """
    )
    
    parser.add_argument("--model_name", type=str, help="Model architecture name")
    parser.add_argument("--type_model", type=str, help="Model type")
    parser.add_argument("--run_time", type=str, required=True, help="Training run timestamp (REQUIRED)")
    parser.add_argument("--dataset_name", type=str, help="Dataset name")
    parser.add_argument("--num_experts", type=int, help="Number of experts")
    parser.add_argument("--top_k", type=int, help="Number of top experts to select")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--router_mode", type=str, choices=["context_aware", "noisy"], 
                       help="Router mode for MoE gating")
    parser.add_argument("--use_context", action="store_true", default=True, 
                       help="Use context features (default: True)")
    parser.add_argument("--no_context", action="store_false", dest="use_context", 
                       help="Disable context features")
    
    return parser.parse_args()


# ============================================================================
# Data Loading & Utilities
# ============================================================================

def setup_test_dataloader(use_context: bool) -> Tuple[DataLoader, object]:
    """Setup test data loader."""
    try:
        logger.info("Loading test dataset...")
        _, _, test_dataset = build_datasets(use_context=use_context)
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.batch_size,
            shuffle=Config.shuffle_test,
            num_workers=0,
            pin_memory='cuda' in Config.device
        )
        
        logger.info(f"Dataset loaded: {len(test_dataset)} samples")
        return test_loader, test_dataset
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise RuntimeError(f"Dataset loading failed: {e}") from e


def get_class_names(test_dataset: object) -> List[str]:
    """Extract class names from dataset."""
    try:
        class_names = [test_dataset.idx_to_class[i] for i in range(len(test_dataset.idx_to_class))]
        logger.info(f"Found {len(class_names)} disease classes")
        return class_names
    except (AttributeError, KeyError) as e:
        logger.error(f"Failed to extract class names: {e}")
        raise


# ============================================================================
# Model Loading
# ============================================================================

def create_model(num_classes: int, num_experts: int, top_k: int, context_dim: int, 
                router_mode: str, temperature: float) -> MoEModel:
    """Create MoE model instance."""
    try:
        logger.info(f"Creating MoE model: experts={num_experts}, top_k={top_k}, mode={router_mode}, temp={temperature}")
        model = MoEModel(
            context_dim=context_dim,
            num_classes=num_classes,
            num_experts=num_experts,
            top_k=top_k,
            router_mode=router_mode,
            temperature=temperature
        )
        model = model.to(Config.device)
        logger.info(f"Model created on {Config.device}")
        return model
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        raise RuntimeError(f"Model creation failed: {e}") from e


def extract_checkpoint_metadata(checkpoint_path: Path) -> Dict:
    """Extract model hyperparameters from checkpoint."""
    try:
        logger.info(f"Loading checkpoint metadata...")
        checkpoint = torch.load(checkpoint_path, map_location=Config.device)
        
        metadata = {
            'num_classes': Config.num_classes,
            'num_experts': Config.num_experts,
            'context_dim': Config.context_dim,
            'top_k': Config.top_k,
            'router_mode': Config.router_mode,
            'temperature': Config.temperature
        }
        
        # Override with checkpoint values if available
        if isinstance(checkpoint, dict):
            for key in ['num_classes', 'num_experts', 'context_dim', 'top_k', 'router_mode', 'temperature']:
                if key in checkpoint:
                    metadata[key] = checkpoint[key]
        
        logger.info(f"Checkpoint: classes={metadata['num_classes']}, experts={metadata['num_experts']}, "
                   f"context_dim={metadata['context_dim']}, top_k={metadata['top_k']}, "
                   f"router_mode={metadata['router_mode']}, temperature={metadata['temperature']}")
        return metadata
    except Exception as e:
        logger.error(f"Failed to extract checkpoint metadata: {e}")
        raise RuntimeError(f"Checkpoint metadata extraction failed: {e}") from e


def load_checkpoint(model: MoEModel, checkpoint_path: Path) -> MoEModel:
    """Load model weights from checkpoint."""
    try:
        logger.info(f"Loading model weights...")
        checkpoint = torch.load(checkpoint_path, map_location=Config.device)
        
        # Handle multiple checkpoint formats
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict)
        model = model.to(Config.device)
        model.eval()
        logger.info("Model weights loaded and set to eval mode")
        return model
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise RuntimeError(f"Checkpoint loading failed: {e}") from e


# ============================================================================
# Inference
# ============================================================================

def run_inference(model: MoEModel, test_loader: DataLoader, use_context: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Perform inference on test dataset."""
    all_preds, all_labels = [], []
    
    try:
        logger.info("Running inference...")
        with torch.inference_mode():
            for batch_idx, batch in enumerate(test_loader):
                if use_context:
                    images, labels, context = batch
                    images, labels, context = images.to(Config.device), labels.to(Config.device), context.to(Config.device) if context is not None else None
                    logits, _, _ = model(images, context)
                else:
                    images, labels = batch
                    images, labels = images.to(Config.device), labels.to(Config.device)
                    logits, _, _ = model(images)
                
                preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                
                if (batch_idx + 1) % 10 == 0:
                    logger.debug(f"Processed {(batch_idx + 1) * Config.batch_size} samples")
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        acc = (all_preds == all_labels).mean()
        logger.info(f"Inference completed: {len(all_labels)} samples, accuracy={acc:.4f}")
        return all_preds, all_labels
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise RuntimeError(f"Inference failed: {e}") from e


# ============================================================================
# Reporting
# ============================================================================

def generate_classification_report(all_labels: np.ndarray, all_preds: np.ndarray, 
                                  target_names: List[str]) -> Dict:
    """Generate classification report."""
    logger.info("Generating classification report...")
    report_text = classification_report(all_labels, all_preds, target_names=target_names)
    print("\n" + "=" * 80)
    print("CLASSIFICATION REPORT")
    print("=" * 80)
    print(report_text)
    return classification_report(all_labels, all_preds, target_names=target_names, output_dict=True)


def save_confusion_matrix(all_labels: np.ndarray, all_preds: np.ndarray, 
                         target_names: List[str], report_dir: Path) -> None:
    """Generate and save confusion matrix visualization."""
    try:
        logger.info("Saving confusion matrix...")
        cm = confusion_matrix(all_labels, all_preds)
        
        plt.figure(figsize=Config.confusion_matrix_figsize)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
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


def save_classification_report_heatmap(report_dict: Dict, target_names: List[str], report_dir: Path) -> None:
    """Generate and save classification report heatmap."""
    try:
        logger.info("Saving classification report heatmap...")
        df = pd.DataFrame(report_dict).transpose()
        
        plt.figure(figsize=Config.classification_report_figsize)
        sns.heatmap(df.iloc[:-1, :-1], annot=True, cmap="Blues", fmt=".2f", cbar_kws={"label": "Score"})
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
# Main
# ============================================================================

def main():
    """Main inference pipeline orchestration."""
    try:
        logger.info("=" * 80)
        logger.info("Starting MoE Model Evaluation")
        logger.info("=" * 80)
        
        # Parse and configure
        args = parse_arguments()
        Config.update_from_args(args)
        
        # Load data
        test_loader, test_dataset = setup_test_dataloader(args.use_context)
        target_names = get_class_names(test_dataset)
        
        # Load checkpoint & extract metadata
        checkpoint_path = Config.get_checkpoint_path()
        metadata = extract_checkpoint_metadata(checkpoint_path)
        Config.num_classes = metadata['num_classes']
        Config.num_experts = metadata['num_experts']
        Config.context_dim = metadata['context_dim']
        Config.top_k = metadata['top_k']
        Config.router_mode = metadata['router_mode']
        Config.temperature = metadata['temperature']
        
        # Create and load model
        model = create_model(
            num_classes=metadata['num_classes'],
            num_experts=metadata['num_experts'],
            top_k=metadata['top_k'],
            context_dim=metadata['context_dim'],
            router_mode=metadata['router_mode'],
            temperature=metadata['temperature']
        )
        model = load_checkpoint(model, checkpoint_path)
        
        # Inference & reporting
        all_preds, all_labels = run_inference(model, test_loader, args.use_context)
        report_dict = generate_classification_report(all_labels, all_preds, target_names)
        report_dir = Config.get_report_dir()
        save_confusion_matrix(all_labels, all_preds, target_names, report_dir)
        save_classification_report_heatmap(report_dict, target_names, report_dir)
        
        logger.info("=" * 80)
        logger.info(f"Evaluation completed successfully")
        logger.info(f"Reports saved to: {report_dir}")
        logger.info("=" * 80)
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
