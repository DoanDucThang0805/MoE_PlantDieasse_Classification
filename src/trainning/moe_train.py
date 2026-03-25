"""
Script Huấn Luyện Mô Hình MoE
============================
Module này thiết lập và thực thi quá trình huấn luyện mô hình Mixture of Experts (MoE)
trên bộ dữ liệu PlantDoc để phân loại bệnh thực vật.
"""

from pathlib import Path
import numpy as np
import warnings

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight

from utils.moe_trainer import MoETrainer
from dataset.plantdoc_dataset import train_dataset, validation_dataset
from models.moe.model import MoEModel
from loss.loss_fn import MoELoss

# Tắt cảnh báo để output sạch hơn
warnings.filterwarnings("ignore")

# ============================================================================
# Các Hằng Số Cấu Hình
# ============================================================================

# Tham số tải dữ liệu
BATCH_SIZE = 32
SHUFFLE_TRAIN = True
SHUFFLE_VAL = False

# Tham số kiến trúc mô hình
NUM_EXPERTS = 8
TOP_K = 4  # Số lượng expert được chọn cho mỗi input

# Siêu tham số huấn luyện
NUM_EPOCHS = 300
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001
MOE_LOSS_ALPHA = 0.05  # Hệ số cân bằng cho hàm loss phụ

# ============================================================================
# Cấu Hình Thiết Bị và Đường Dẫn
# ============================================================================

# Xác định thiết bị (GPU nếu có sẵn, nếu không dùng CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Thiết lập thư mục checkpoint để lưu trọng số mô hình
output_dir = Path.cwd().parents[0]
checkpoint_dir = output_dir / "checkpoints" / "plantdoc" / "MoE" / "mobilenetv3large_moe"

# ============================================================================
# Chuẩn Bị Dữ Liệu
# ============================================================================

# Tạo data loader cho tập huấn luyện và kiểm định
train_ds = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE_TRAIN
)
val_ds = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE_VAL
)

# Trích xuất nhãn và xác định số lượng lớp
labels = train_dataset.labels
num_classes = len(set(labels))
print(f"Số lượng lớp: {num_classes}")

# Tính toán trọng số lớp cân bằng để xử lý mất cân bằng dữ liệu
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(num_classes),
    y=labels
)

# ============================================================================
# Khởi Tạo Mô Hình, Loss Function và Optimizer
# ============================================================================

# Khởi tạo mô hình MoE với kiến trúc được chỉ định
model = MoEModel(
    num_classes=num_classes,
    num_experts=NUM_EXPERTS,
    top_k=TOP_K
)

# Tạo hàm loss với cân bằng loss phụ
criterion = MoELoss(alpha=MOE_LOSS_ALPHA)

# Thiết lập optimizer để tối ưu hóa tham số mô hình
optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# ============================================================================
# Thiết Lập Huấn Luyện
# ============================================================================

# Khởi tạo trainer với tất cả các thành phần
trainer = MoETrainer(
    num_epochs=NUM_EPOCHS,
    device=device,
    train_loader=train_ds,
    val_loader=val_ds,
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    batch_size=BATCH_SIZE,
    checkpoint_dir=checkpoint_dir
)


if __name__ == "__main__":
    # Thực thi quy trình huấn luyện
    trainer.train()
