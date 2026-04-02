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
    python context_moe_train.py --router_mode context_aware --use_context True

Các tham số:
    --router_mode: Chế độ router cho MoE gating ('noisy' hoặc 'context_aware')
    --use_context: Có sử dụng context features hay không (True/False)

Tác giả: MoE Team
Phiên bản: 1.0
"""

from pathlib import Path
import numpy as np
import argparse
import warnings

import torch
from torchinfo import summary
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight

from utils.context_moe_trainner import ContextAwareMoETrainer
from dataset.plantdoc_dataset import build_datasets
from models.moe.model import MoEModel
from loss.loss_fn import MoELoss

# Tắt cảnh báo để output sạch hơn
warnings.filterwarnings("ignore")


# ============================================================================
# Cấu Hình Mô Hình
# ============================================================================

class Config:
    """
    Quản lý toàn bộ cấu hình cho quá trình huấn luyện mô hình MoE.
    
    Attributes:
        batch_size (int): Kích thước batch cho huấn luyện (mặc định: 32)
        shuffle_train (bool): Có xáo trộn dữ liệu huấn luyện hay không (mặc định: True)
        shuffle_val (bool): Có xáo trộn dữ liệu kiểm định hay không (mặc định: False)
        num_experts (int): Số lượng experts trong mô hình MoE (mặc định: 8)
        top_k (int): Số lượng experts được chọn cho mỗi input (mặc định: 4)
        context_feature_dim (int): Chiều của context features (mặc định: 6)
        num_epochs (int): Số epoch để huấn luyện (mặc định: 300)
        learning_rate (float): Tốc độ học cho optimizer (mặc định: 0.001)
        weight_decay (float): Hệ số weight decay (mặc định: 0.001)
        moe_loss_alpha (float): Hệ số cân bằng cho hàm loss phụ (mặc định: 0.05)
        device (str): Thiết bị để chạy mô hình ('cuda' hoặc 'cpu')
        checkpoint_parent (str): Thư mục cha cho checkpoints (mặc định: 'checkpoints')
        checkpoint_subdir (str): Đường dẫn con cho checkpoints
    
    Methods:
        get_checkpoint_dir(): Lấy đường dẫn thư mục checkpoint và tạo nó nếu chưa tồn tại
    """
    
    # Tham số tải dữ liệu
    batch_size = 32
    shuffle_train = True
    shuffle_val = False
    
    # Tham số kiến trúc mô hình
    num_experts = 6
    top_k = 2  # Số lượng expert được chọn cho mỗi input
    context_feature_dim = 6
    
    # Siêu tham số huấn luyện
    num_epochs = 300
    learning_rate = 0.001
    weight_decay = 0.001
    moe_loss_alpha = 0.05  # Hệ số cân bằng cho hàm loss phụ
    
    # Tham số thiết bị
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Đường dẫn
    checkpoint_parent = "checkpoints"
    checkpoint_subdir = "plantdoc/MoE/mobilenetv3large_moe"
    
    @classmethod
    def get_checkpoint_dir(cls) -> Path:
        """
        Lấy đường dẫn thư mục checkpoint và tạo nó nếu chưa tồn tại.
        
        Hàm này xác định vị trí của thư mục checkpoint dựa trên cấu hình được thiết lập
        và tự động tạo thư mục nếu nó chưa tồn tại.
        
        Returns:
            Path: Đối tượng Path đến thư mục checkpoint
            
        Raises:
            PermissionError: Nếu không có quyền tạo thư mục
            OSError: Nếu có lỗi hệ thống khi tạo thư mục
            
        Example:
            >>> checkpoint_dir = Config.get_checkpoint_dir()
            >>> print(checkpoint_dir)
            /path/to/checkpoints/plantdoc/MoE/mobilenetv3large_moe
        """
        output_dir = Path.cwd().parents[0]
        checkpoint_dir = output_dir / cls.checkpoint_parent / cls.checkpoint_subdir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir


def parse_arguments() -> argparse.Namespace:
    """
    Phân tích các tham số dòng lệnh.
    
    Hàm này xử lý các tham số được truyền từ dòng lệnh khi chạy script.
    
    Returns:
        argparse.Namespace: Đối tượng chứa các tham số được phân tích
            - router_mode (str): Chế độ router ('noisy' hoặc 'context_aware')
            - use_context (bool): Có sử dụng context features hay không
            
    Example:
        >>> args = parse_arguments()
        >>> print(args.router_mode)
        'context_aware'
        >>> print(args.use_context)
        True
    """
    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình MoE cho phân loại bệnh thực vật"
    )
    parser.add_argument(
        "--router_mode",
        type=str,
        default="noisy",
        choices=["noisy", "context_aware"],
        help="Router mode for MoE gating"
    )
    parser.add_argument(
        "--use_context",
        type=bool,
        default=True,
        choices=[True, False],
        help="Whether to use context features in the MoE model"
    )
    return parser.parse_args()


def setup_dataloaders(use_context: bool) -> tuple:
    """
    Chuẩn bị các DataLoader cho tập huấn luyện, kiểm định và kiểm thử.
    
    Hàm này thực hiện load tập dữ liệu, tạo DataLoaders, tính toán trọng số lớp
    cân bằng, và xác định số lượng lớp phân loại.
    
    Args:
        use_context (bool): Có sử dụng context features hay không
    
    Returns:
        tuple: Tuple chứa:
            - train_loader (DataLoader): DataLoader cho tập huấn luyện
            - val_loader (DataLoader): DataLoader cho tập kiểm định
            - num_classes (int): Số lượng lớp phân loại
            
    Note:
        - Tập kiểm định không được xáo trộn (shuffle=False)
        - Tập huấn luyện được xáo trộn (shuffle=True)
        - In ra số lượng lớp được phát hiện
        
    Example:
        >>> train_loader, val_loader, num_classes = setup_dataloaders(True)
        >>> print(f"Số lượng lớp: {num_classes}")
        Số lượng lớp: 10
    """
    train_dataset, validation_dataset, test_dataset = build_datasets(use_context)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=Config.shuffle_train
    )
    
    val_loader = DataLoader(
        validation_dataset,
        batch_size=Config.batch_size,
        shuffle=Config.shuffle_val
    )
    
    labels = train_dataset.labels
    num_classes = len(set(labels))
    print(f"Số lượng lớp: {num_classes}")
    
    # Tính toán trọng số lớp cân bằng
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=labels
    )
    
    return train_loader, val_loader, num_classes


def create_model(num_classes: int, router_mode: str) -> MoEModel:
    """
    Tạo và khởi tạo mô hình MoE.
    
    Hàm này khởi tạo mô hình Mixture of Experts với các cấu hình được chỉ định.
    Mô hình bao gồm nhiều experts và một gating network để chọn experts.
    
    Args:
        num_classes (int): Số lượng lớp phân loại
        router_mode (str): Chế độ router, có thể là 'noisy' hoặc 'context_aware'
    
    Returns:
        MoEModel: Mô hình MoE được khởi tạo với các tham số từ Config
        
    Raises:
        ValueError: Nếu router_mode không hợp lệ
        
    Example:
        >>> model = create_model(num_classes=10, router_mode='context_aware')
        >>> print(f"Tổng tham số mô hình: {sum(p.numel() for p in model.parameters())}")
    
    Note:
        Sử dụng cấu hình từ Config class:
        - context_feature_dim: Chiều của context features
        - num_experts: Số lượng experts
        - top_k: Số experts được chọn cho mỗi input
    """
    model = MoEModel(
        context_dim=Config.context_feature_dim,
        num_classes=num_classes,
        num_experts=Config.num_experts,
        top_k=Config.top_k,
        router_mode=router_mode
    )
    if model.router_mode == "context_aware":
        summary(model, input_data=(torch.randn(1, 3, 224, 224), torch.randn(1, Config.context_feature_dim)), col_names=["input_size", "output_size", "num_params", "trainable"])
    else:
        summary(model, input_data=torch.randn(1, 3, 224, 224), col_names=["input_size", "output_size", "num_params", "trainable"])
    return model


def setup_training_components(model: MoEModel) -> tuple:
    """
    Thiết lập hàm loss và optimizer cho huấn luyện.
    
    Hàm này khởi tạo MoELoss (hàm loss tùy chỉnh cho MoE) và optimizer Adam
    để tối ưu hóa các tham số mô hình.
    
    Args:
        model (MoEModel): Mô hình cần tối ưu hóa
    
    Returns:
        tuple: Tuple chứa:
            - criterion (MoELoss): Hàm loss cho MoE
            - optimizer (optim.Adam): Optimizer Adam
            
    Note:
        - Loss function bao gồm hệ số cân bằng (alpha) từ Config
        - Optimizer sử dụng learning_rate và weight_decay từ Config
        
    Example:
        >>> model = create_model(10, 'context_aware')
        >>> criterion, optimizer = setup_training_components(model)
        >>> print(f"Optimizer: {optimizer.__class__.__name__}")
        Optimizer: Adam
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
    optimizer
) -> ContextAwareMoETrainer:
    """
    Tạo trainer để thực thi quá trình huấn luyện.
    
    Hàm này khởi tạo ContextAwareMoETrainer với tất cả các thành phần cần thiết
    để bắt đầu quá trình huấn luyện mô hình.
    
    Args:
        train_loader (DataLoader): DataLoader cho tập huấn luyện
        val_loader (DataLoader): DataLoader cho tập kiểm định
        model (MoEModel): Mô hình để huấn luyện
        criterion: Hàm loss để tính toán sai số
        optimizer: Optimizer để cập nhật tham số mô hình
    
    Returns:
        ContextAwareMoETrainer: Trainer object được khởi tạo sẵn sàng cho huấn luyện
        
    Note:
        - Trainer sẽ tự động tạo thư mục checkpoint nếu chưa tồn tại
        - Sử dụng các tham số từ Config class (num_epochs, device, batch_size)
        
    Example:
        >>> trainer = create_trainer(train_loader, val_loader, model, criterion, optimizer)
        >>> trainer.train()  # Bắt đầu huấn luyện
    """
    checkpoint_dir = Config.get_checkpoint_dir()
    
    trainer = ContextAwareMoETrainer(
        num_epochs=Config.num_epochs,
        device=Config.device,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        batch_size=Config.batch_size,
        checkpoint_dir=checkpoint_dir
    )
    
    return trainer


def main():
    """
    Hàm chính: điều phối toàn bộ quá trình huấn luyện.
    
    Hàm này là điểm vào chính của script. Nó điều phối tất cả các bước huấn luyện:
    1. Phân tích tham số dòng lệnh
    2. Chuẩn bị dữ liệu (load datasets, tạo DataLoaders)
    3. Tạo mô hình MoE
    4. Thiết lập loss function và optimizer
    5. Tạo trainer
    6. Thực thi quá trình huấn luyện
    
    Flow:
        parse_arguments() -> setup_dataloaders() -> create_model() ->
        setup_training_components() -> create_trainer() -> trainer.train()
    
    Returns:
        None
        
    Note:
        - Mô hình sẽ được di chuyển đến device được chỉ định (GPU/CPU)
        - Checkpoints sẽ được lưu trong quá trình huấn luyện
        - Quá trình huấn luyện có thể mất thời gian tùy thuộc vào dữ liệu và cấu hình
        
    Example:
        Để chạy script từ dòng lệnh:
        >>> python context_moe_train.py --router_mode context_aware --use_context True
    """
    # Phân tích tham số
    args = parse_arguments()
    
    # Chuẩn bị dữ liệu
    train_loader, val_loader, num_classes = setup_dataloaders(args.use_context)
    
    # Tạo mô hình
    model = create_model(num_classes, args.router_mode)
    model.to(Config.device)
    
    # Thiết lập loss function và optimizer
    criterion, optimizer = setup_training_components(model)
    
    # Tạo trainer
    trainer = create_trainer(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer
    )
    
    print("======== Training Config ========")
    print(f"Router mode: {args.router_mode}")
    print(f"Use context: {args.use_context}")
    print(f"Experts: {Config.num_experts}")
    print(f"Top-k: {Config.top_k}")
    print(f"Epochs: {Config.num_epochs}")
    print(f"Device: {Config.device}")
    print("=================================")

    # Thực thi huấn luyện
    trainer.train()


if __name__ == "__main__":
    """
    Entry point của script.
    
    Chạy hàm main() khi script được chạy trực tiếp (không được import như module).
    """
    main()
