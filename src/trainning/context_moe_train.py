"""
Script Huấn Luyện Mô Hình MoE
============================

Module này thiết lập và thực thi quá trình huấn luyện mô hình Mixture of Experts (MoE)
trên bộ dữ liệu PlantDoc để phân loại bệnh thực vật.

Tổng quan:
    - Tạo và huấn luyện mô hình MoE với gating router
    - Hỗ trợ hai chế độ router: 'noisy' và 'context_aware'
    - Sử dụng context features để cải thiện hiệu suất phân loại
    - Lưu checkpoint mô hình tốt nhất trong quá trình huấn luyện

Cách sử dụng:
    python context_moe_train.py \
        --batch_size 32 \
        --epochs 300 \
        --num_experts 8 \
        --top_k 4 \
        --model_name mobilenetv3large_moe \
        --type_model MoE \
        --router_mode context_aware \
        --use_context True

Các tham số:
    --batch_size: Kích thước batch cho huấn luyện (mặc định: 32)
    --epochs: Số epoch để huấn luyện (mặc định: 300)
    --num_experts: Số lượng experts trong mô hình MoE (mặc định: 8)
    --top_k: Số lượng experts được chọn cho mỗi input (mặc định: 4)
    --model_name: Tên kiến trúc mô hình (mặc định: mobilenetv3large_moe)
    --type_model: Loại mô hình (mặc định: MoE)
    --router_mode: Chế độ router cho MoE gating ('noisy' hoặc 'context_aware')
    --use_context: Có sử dụng context features hay không (True/False)

Tác giả: MoE Team
Phiên bản: 2.0
"""

from pathlib import Path
import argparse
import warnings
import logging

import torch
from torchinfo import summary
from torch.utils.data import DataLoader
import torch.optim as optim

from utils.context_moe_trainner import ContextAwareMoETrainer
from dataset.plantdoc_dataset import build_datasets
from models.moe.model import MoEModel
from loss.loss_fn import MoELoss

# Tắt cảnh báo để output sạch hơn
warnings.filterwarnings("ignore")

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
    Dynamic configuration management for MoE model training.
    
    Configuration values are set from command-line arguments,
    allowing flexible model training with different parameters.
    
    Attributes:
        batch_size (int): Batch size for training
        num_epochs (int): Number of training epochs
        num_experts (int): Number of experts in MoE model
        top_k (int): Number of experts selected per input
        model_name (str): Name of the model architecture
        type_model (str): Type of model ('MoE', 'pretrained', etc.)
        context_feature_dim (int): Dimension of context features
        shuffle_train (bool): Whether to shuffle training data
        shuffle_val (bool): Whether to shuffle validation data
        learning_rate (float): Learning rate for optimizer
        weight_decay (float): Weight decay for optimizer
        moe_loss_alpha (float): Balance coefficient for auxiliary loss
        device (str): Computing device ('cuda' or 'cpu')
        checkpoint_parent (str): Parent directory for checkpoints
    
    Methods:
        get_checkpoint_dir(): Get checkpoint directory path
        update_from_args(): Update config from argparse arguments
    """
    
    # Default values
    batch_size: int = 32
    num_epochs: int = 300
    num_experts: int = 6
    top_k: int = 4
    model_name: str = 'mobilenetv3large_moe'
    type_model: str = 'MoE'
    
    # Fixed parameters
    context_feature_dim: int = 6
    use_context: bool = True
    shuffle_train: bool = True
    shuffle_val: bool = False
    learning_rate: float = 0.001
    weight_decay: float = 0.001
    moe_loss_alpha: float = 0.05
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_parent: str = "checkpoints"
    

    @classmethod
    def update_from_args(cls, args: argparse.Namespace) -> None:
        """
        Update configuration from command-line arguments.
        
        All CLI arguments automatically update corresponding Config class attributes.
        Example: --batch_size 64 → Config.batch_size = 64
        
        Args:
            args (argparse.Namespace): Parsed command-line arguments
        """
        cls.batch_size = args.batch_size
        cls.num_epochs = args.epochs
        cls.num_experts = args.num_experts
        cls.top_k = args.top_k
        cls.model_name = args.model_name
        cls.type_model = args.type_model
        cls.use_context = args.use_context

    @classmethod
    def get_checkpoint_dir(cls) -> Path:
        """
        Get checkpoint directory path and create if not exists.
        
        Returns:
            Path: Path to checkpoint directory
            
        Raises:
            PermissionError: If no permission to create directory
            OSError: If system error when creating directory
        """
        output_dir = Path.cwd().parents[0]
        checkpoint_subdir = f"plantdoc/{cls.type_model}/{cls.model_name}"
        checkpoint_dir = output_dir / cls.checkpoint_parent / checkpoint_subdir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoint directory: {checkpoint_dir}")
        return checkpoint_dir


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for training configuration.
    
    Defines and parses all CLI arguments for flexible model training.
    
    Returns:
        argparse.Namespace: Parsed arguments containing:
            - batch_size (int): Batch size for training
            - epochs (int): Number of training epochs
            - num_experts (int): Number of experts in MoE
            - top_k (int): Number of experts selected per input
            - model_name (str): Model architecture name
            - type_model (str): Model type
            - router_mode (str): Router mode for MoE gating
            - use_context (bool): Whether to use context features
            
    Example:
        >>> args = parse_arguments()
        >>> print(f"Batch size: {args.batch_size}, Epochs: {args.epochs}")
    """
    parser = argparse.ArgumentParser(
        description="Train MoE model for plant disease classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    python context_moe_train.py --batch_size 32 --epochs 300
    python context_moe_train.py --num_experts 8 --top_k 4 --model_name mobilenetv3large_moe
    python context_moe_train.py --batch_size 64 --epochs 200 --num_experts 6 --top_k 2
            """
    )
    
    # Data loading parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)"
    )
    
    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of epochs for training (default: 300)"
    )
    
    # Model architecture parameters
    parser.add_argument(
        "--num_experts",
        type=int,
        default=6,
        help="Number of experts in MoE model (default: 6)"
    )
    
    parser.add_argument(
        "--top_k",
        type=int,
        default=2,
        help="Number of experts selected per input (default: 2)"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="mobilenetv3large_moe",
        help="Model architecture name (default: mobilenetv3large_moe)"
    )
    
    parser.add_argument(
        "--type_model",
        type=str,
        default="MoE",
        choices=["MoE", "pretrained", "other"],
        help="Model type (default: MoE)"
    )
    
    # Router configuration
    parser.add_argument(
        "--router_mode",
        type=str,
        default="context_aware",
        choices=["noisy", "context_aware"],
        help="Router mode for MoE gating (default: context_aware)"
    )
    
    # Context features
    parser.add_argument(
        "--use_context",
        type=bool,
        default=True,
        choices=[True, False],
        help="Whether to use context features (default: True)"
    )
    
    return parser.parse_args()


def setup_dataloaders(use_context: bool, batch_size: int) -> tuple:
    """
    Setup DataLoaders for training, validation and test sets.
    
    Loads datasets, creates DataLoaders, computes class weights,
    and determines number of classification classes.
    
    Args:
        use_context (bool): Whether to load context features
        batch_size (int): Batch size for DataLoaders
    
    Returns:
        tuple: Tuple containing:
            - train_loader (DataLoader): DataLoader for training set
            - val_loader (DataLoader): DataLoader for validation set
            - num_classes (int): Number of classification classes
            
    Note:
        - Validation set is not shuffled (shuffle=False)
        - Training set is shuffled (shuffle=True)
        - Logs detected number of classes
        
    Example:
        >>> train_loader, val_loader, num_classes = setup_dataloaders(True, 32)
        >>> print(f"Number of classes: {num_classes}")
    """
    train_dataset, validation_dataset, _ = build_datasets(use_context)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=Config.shuffle_train
    )
    
    val_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=Config.shuffle_val
    )
    
    labels = train_dataset.labels
    num_classes = len(set(labels))
    logger.info(f"Number of classes detected: {num_classes}")
    
    return train_loader, val_loader, num_classes


def _display_model_summary(model: MoEModel, router_mode: str) -> None:
    """
    Display model architecture summary (helper function).
    
    Args:
        model (MoEModel): Model to summarize
        router_mode (str): Router mode ('context_aware' or 'noisy')
    """
    if router_mode == "context_aware":
        summary(model, input_data=(torch.randn(1, 3, 224, 224), torch.randn(1, Config.context_feature_dim)), 
                col_names=["input_size", "output_size", "num_params", "trainable"])
    else:
        summary(model, input_data=torch.randn(1, 3, 224, 224), 
                col_names=["input_size", "output_size", "num_params", "trainable"])


def create_model(num_classes: int, router_mode: str, num_experts: int, top_k: int) -> MoEModel:
    """
    Create and initialize MoE model architecture.
    
    Initializes a Mixture of Experts model with specified configuration.
    Model includes multiple experts and a gating network for expert selection.
    
    Args:
        num_classes (int): Number of classification classes
        router_mode (str): Router mode, either 'noisy' or 'context_aware'
        num_experts (int): Number of experts in MoE
        top_k (int): Number of experts selected per input
    
    Returns:
        MoEModel: Initialized MoE model with Config parameters
        
    Raises:
        RuntimeError: If model creation fails
    """
    try:
        logger.info(f"Creating MoE model with {num_experts} experts, top_k={top_k}")
        
        model = MoEModel(
            context_dim=Config.context_feature_dim,
            num_classes=num_classes,
            num_experts=num_experts,
            top_k=top_k,
            router_mode=router_mode,
            use_context=Config.use_context
        )
        
        _display_model_summary(model, router_mode)
        return model
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        raise RuntimeError(f"Model creation failed: {e}") from e


def setup_training_components(model: MoEModel) -> tuple:
    """
    Setup loss function and optimizer for training.
    
    Initializes MoELoss (custom loss for MoE) and Adam optimizer
    to optimize model parameters.
    
    Args:
        model (MoEModel): Model to optimize
    
    Returns:
        tuple: Tuple containing:
            - criterion (MoELoss): Loss function for MoE
            - optimizer (optim.Adam): Adam optimizer
            
    Note:
        - Loss function includes balance coefficient (alpha) from Config
        - Optimizer uses learning_rate and weight_decay from Config
    """
    criterion = MoELoss(alpha=Config.moe_loss_alpha)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay
    )
    
    return criterion, optimizer


def create_trainer(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: MoEModel,
    criterion,
    optimizer,
    num_epochs: int,
    batch_size: int
) -> ContextAwareMoETrainer:
    """
    Create trainer for model training execution.
    
    Initializes ContextAwareMoETrainer with all necessary components
    for starting the model training process.
    
    Args:
        train_loader (DataLoader): DataLoader for training set
        val_loader (DataLoader): DataLoader for validation set
        model (MoEModel): Model to train
        criterion: Loss function for error computation
        optimizer: Optimizer for parameter updates
        num_epochs (int): Number of epochs for training
        batch_size (int): Batch size for training
    
    Returns:
        ContextAwareMoETrainer: Trainer object ready for training execution
        
    Note:
        - Trainer will automatically create checkpoint directory if not exists
        - Uses device from Config class
        
    Example:
        >>> trainer = create_trainer(train_loader, val_loader, model, criterion, optimizer, 300, 32)
        >>> trainer.train()  # Start training
    """
    checkpoint_dir = Config.get_checkpoint_dir()
    
    trainer = ContextAwareMoETrainer(
        num_epochs=num_epochs,
        device=Config.device,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        batch_size=batch_size,
        checkpoint_dir=checkpoint_dir
    )
    
    return trainer


def main():
    """
    Main execution function orchestrating the entire training pipeline.
    
    Function is the main entry point of the script. It orchestrates all training steps:
    1. Parse command-line arguments
    2. Update Config from arguments
    3. Prepare data (load datasets, create DataLoaders)
    4. Create MoE model
    5. Setup loss function and optimizer
    6. Create trainer
    7. Execute training process
    
    Flow:
        parse_arguments() -> Config.update_from_args() -> setup_dataloaders() ->
        create_model() -> setup_training_components() -> create_trainer() -> trainer.train()
    
    Returns:
        None
        
    Note:
        - Model will be moved to specified device (GPU/CPU)
        - Checkpoints will be saved during training
        - Training time depends on data and configuration
        
    Example:
        Running script from command line:
        >>> python context_moe_train.py --batch_size 32 --epochs 300 --num_experts 8 --top_k 4
    """
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Update Config from arguments
        Config.update_from_args(args)
        
        # Log configuration
        logger.info("=" * 80)
        logger.info("Starting MoE Model Training")
        logger.info("=" * 80)
        logger.info(
            f"\nTraining Configuration:"
            f"\n  Batch Size: {Config.batch_size}"
            f"\n  Epochs: {Config.num_epochs}"
            f"\n  Num Experts: {Config.num_experts}"
            f"\n  Top-K: {Config.top_k}"
            f"\n  Model: {Config.model_name} ({Config.type_model})"
            f"\n  Router Mode: {args.router_mode}"
            f"\n  Use Context: {args.use_context}"
            f"\n  Device: {Config.device}"
            f"\n  Learning Rate: {Config.learning_rate}"
            f"\n  Weight Decay: {Config.weight_decay}"
            f"\n  MoE Loss Alpha: {Config.moe_loss_alpha}"
        )
        logger.info("=" * 80)
        
        # Setup data
        train_loader, val_loader, num_classes = setup_dataloaders(args.use_context, Config.batch_size)
        
        # Create model
        model = create_model(
            num_classes=num_classes,
            router_mode=args.router_mode,
            num_experts=Config.num_experts,
            top_k=Config.top_k
        )
        model.to(Config.device)
        
        # Setup loss function and optimizer
        criterion, optimizer = setup_training_components(model)
        
        # Create trainer
        trainer = create_trainer(
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=Config.num_epochs,
            batch_size=Config.batch_size
        )
        
        # Execute training
        trainer.train()
        
        logger.info("=" * 80)
        logger.info("Training Completed Successfully")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.critical(f"Fatal error during training: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
