"""
Script Đánh Giá và Suy Luận Mô Hình MoE
======================================
Module này thực hiện suy luận trên bộ dữ liệu kiểm tra bằng mô hình đã huấn luyện
và tạo ra các báo cáo đánh giá toàn diện bao gồm số liệu phân loại, ma trận nhầm lẫn
và biểu đồ hiệu suất.

Tính năng:
    - Tải checkpoint mô hình đã huấn luyện
    - Suy luận batch trên dữ liệu kiểm tra
    - Tạo báo cáo phân loại
    - Hình dung ma trận nhầm lẫn
    - Tạo bản đồ hiệu suất
"""

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from dataset.plantdoc_dataset import test_dataset
from models.pretrained_model.mobilenetv3_large import model


# ============================================================================
# Cấu Hình Chính
# ============================================================================

# Thông tin mô hình và checkpoint
MODEL_NAME = 'mobilenetv3_large'
MODEL_TYPE = 'pretrain_weight'
RUN_TIME = 'run_20260126-122233'  # Timestamp của lần chạy huấn luyện
DATASET_NAME = 'plantdoc'

# Tham số tải dữ liệu
BATCH_SIZE = 32
SHUFFLE_TEST = True

# Tham số hình ảnh plot
CONFUSION_MATRIX_FIGSIZE = (12, 10)
CLASSIFICATION_REPORT_FIGSIZE = (10, 6)
REPORT_DPI = 300

# ============================================================================
# Cấu Hình Thiết Bị và Đường Dẫn
# ============================================================================

# Xác định thiết bị (GPU nếu có sẵn, nếu không dùng CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Xác định đường dẫn checkpoint (tương đối với vị trí file này, bao gồm các CWD khác)
checkpoint_path = (
    Path(__file__).resolve().parents[3] / 'checkpoints' / DATASET_NAME / MODEL_TYPE / 
    MODEL_NAME / RUN_TIME / 'best_checkpoint.pth'
)

# Xác định thư mục lưu báo cáo
report_dir = (
    Path(__file__).resolve().parents[3] / 'reports' / DATASET_NAME / MODEL_TYPE / MODEL_NAME / RUN_TIME
)

# ============================================================================
# Tải Dữ Liệu Kiểm Tra
# ============================================================================

# Tạo data loader cho tập kiểm tra
test_ds = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE_TEST
)


# ============================================================================
# Tải Checkpoint
# ============================================================================

# Kiểm tra sự tồn tại của checkpoint
print(f"Đang tải checkpoint từ: {checkpoint_path}")
if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint không tồn tại: {checkpoint_path}")

# Tải checkpoint hỗ trợ nhiều định dạng khác nhau
checkpoint = torch.load(checkpoint_path, map_location=device)

# Hỗ trợ các tên khóa khác nhau trong checkpoint
if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    # Nếu checkpoint chính nó là state_dict, sử dụng trực tiếp
    state_dict = checkpoint

# Tải mô hình lên thiết bị và đặt chế độ đánh giá
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# ============================================================================
# Suy Luận Trên Tập Kiểm Tra
# ============================================================================

# Danh sách lưu trữ tất cả các dự đoán và nhãn thực tế
all_preds = []
all_labels = []

# Suy luận trên từng batch
print("Đang thực hiện suy luận trên tập kiểm tra...")
with torch.inference_mode(True):
    for images, labels in test_ds:
        # Chuyển dữ liệu sang thiết bị
        images, labels = images.to(device), labels.to(device)
        
        # Thực hiện suy luận (mô hình trả về logits, auxiliary loss và expert assignment)
        logits = model(images)
        
        # Tính toán xác suất và dự đoán nhãn
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        # Lưu trữ kết quả
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

print(f"Hoàn tất suy luận. Tổng cộng {len(all_labels)} mẫu được xử lý.")

# ============================================================================
# Tạo Báo Cáo Phân Loại
# ============================================================================

# Lấy tên các lớp từ dataset
target_names = [test_dataset.idx_to_class[i] for i in range(len(test_dataset.idx_to_class))]
print("\nCác lớp bệnh:")
print(target_names)

# In báo cáo phân loại chi tiết
print("\n" + "="*80)
print("BÁNG CÁO PHÂN LOẠI")
print("="*80)
print(classification_report(all_labels, all_preds, target_names=target_names))

# ============================================================================
# Khởi Tạo Thư Mục Lưu Báo Cáo
# ============================================================================

# Tạo thư mục báo cáo nếu chưa tồn tại
report_dir.mkdir(parents=True, exist_ok=True)
print(f"\nSaving báo cáo đến: {report_dir}")

# ============================================================================
# Hình Dung Ma Trận Nhầm Lẫn
# ============================================================================

# Tính toán ma trận nhầm lẫn
cm = confusion_matrix(all_labels, all_preds)

# Tạo biểu đồ ma trận nhầm lẫn
plt.figure(figsize=CONFUSION_MATRIX_FIGSIZE)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_names,
    yticklabels=target_names
)

plt.xlabel("Nhãn Dự Đoán", fontsize=12)
plt.ylabel("Nhãn Thực Tế", fontsize=12)
plt.title("Ma Trận Nhầm Lẫn - Phân Loại Bệnh Lá Cà Chua", fontsize=14)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

# Lưu hình ảnh
plt.savefig(report_dir / "confusion_matrix.png", dpi=REPORT_DPI, bbox_inches="tight")
print(f"✓ Đã lưu ma trận nhầm lẫn: {report_dir / 'confusion_matrix.png'}")
plt.show()

# ============================================================================
# Hình Dung Báo Cáo Phân Loại
# ============================================================================

# Tạo báo cáo phân loại dạng từ điển
report_dict = classification_report(
    all_labels,
    all_preds,
    target_names=target_names,
    output_dict=True
)

# Chuyển đổi thành DataFrame để hình dung
df = pd.DataFrame(report_dict).transpose()

# Tạo biểu đồ nhiệt cho báo cáo phân loại
plt.figure(figsize=CLASSIFICATION_REPORT_FIGSIZE)
sns.heatmap(
    df.iloc[:-1, :-1],
    annot=True,
    cmap="Blues",
    fmt=".2f",
    cbar_kws={"label": "Giá Trị"}
)

plt.title("Báo Cáo Phân Loại (Precision / Recall / F1-score)", fontsize=12)
plt.xlabel("Chỉ Số Đánh Giá")
plt.ylabel("Lớp Bệnh")
plt.tight_layout()

# Lưu hình ảnh
plt.savefig(
    report_dir / "classification_report_heatmap.png",
    dpi=REPORT_DPI,
    bbox_inches="tight"
)
print(f"✓ Đã lưu báo cáo phân loại: {report_dir / 'classification_report_heatmap.png'}")
plt.show()

print("\n" + "="*80)
print("HOÀN TẤT ĐÁNH GIÁ MÔ HÌNH")
print("="*80)
