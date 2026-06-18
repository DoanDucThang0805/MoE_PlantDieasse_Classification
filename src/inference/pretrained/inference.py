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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from dataset.plantdoc_dataset import build_datasets
from dataset.plantdoc_datasetv2 import build_datasets as build_datasets_v2
from models.pretrained_model.mobilenetv3_small import model


# ============================================================================
# Cấu Hình Chính
# ============================================================================

# Thông tin mô hình và checkpoint
MODEL_NAME = 'mobilenetv3_small'  # Tên mô hình (phù hợp với tên thư mục checkpoint)
MODEL_TYPE = 'pretrain_models'
RUN_TIME = 'run_20260515-002446'  # Timestamp của lần chạy huấn luyện
DATASET_NAME = 'plantdoc'
SEED = 44

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
    MODEL_NAME / f'seed_{SEED}' / RUN_TIME / 'best_checkpoint.pth'
)

# Xác định thư mục lưu báo cáo
report_dir = (
    Path(__file__).resolve().parents[3] / 'reports' / DATASET_NAME / MODEL_TYPE / MODEL_NAME /  f'seed_{SEED}' / RUN_TIME
)

# ============================================================================
# Tải Dữ Liệu Kiểm Tra
# ============================================================================

# _, _, test_dataset = build_datasets(False)
_, _, test_dataset = build_datasets(use_context=False)

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
    for images, labels, _ in test_ds:
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
acc = accuracy_score(all_labels, all_preds)
print('Accuracy', acc)
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

"""Model Evaluation and Inference Script
======================================
This module performs inference on test datasets using trained models
and generates comprehensive evaluation reports including classification metrics, confusion matrices,
and performance visualizations.

Features:
    - Load trained model checkpoints
    - Batch inference on test data
    - Generate classification reports
    - Visualize confusion matrices
    - Create performance maps
"""

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from dataset.plantdoc_dataset import build_datasets
from models.pretrained_model.mobilenetv3_small import model


# ============================================================================
# Main Configuration
# ============================================================================

# Model and checkpoint information
MODEL_NAME = 'mobilenetv3_small'  # Model name (matches checkpoint directory name)
MODEL_TYPE = 'pretrain_models'
RUN_TIME = 'run_20260516-031725'  # Timestamp of training run
DATASET_NAME = 'plantdoc'
SEED = 46

# Data loading parameters
BATCH_SIZE = 32
SHUFFLE_TEST = True

# Plot image parameters
CONFUSION_MATRIX_FIGSIZE = (10,8)
CLASSIFICATION_REPORT_FIGSIZE = (10, 6)
REPORT_DPI = 300

# ============================================================================
# Device and Path Configuration
# ============================================================================

# Determine device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Determine checkpoint path (relative to this file location, handles different CWDs)
checkpoint_path = (
    Path(__file__).resolve().parents[3] / 'checkpoints' / DATASET_NAME / MODEL_TYPE / 
    MODEL_NAME / f'seed_{SEED}' / RUN_TIME / 'best_checkpoint.pth'
)

# Determine report directory
report_dir = (
    Path(__file__).resolve().parents[3] / 'reports' / DATASET_NAME / MODEL_TYPE / MODEL_NAME /  f'seed_{SEED}' / RUN_TIME
)

# ============================================================================
# Load Test Data
# ============================================================================

# _, _, test_dataset = build_datasets(False)
_, _, test_dataset = build_datasets(use_context=False)

# Create data loader for test set
test_ds = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE_TEST
)


# ============================================================================
# Load Checkpoint
# ============================================================================

# Check checkpoint existence
print(f"Loading checkpoint from: {checkpoint_path}")
if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

# Load checkpoint supporting multiple formats
checkpoint = torch.load(checkpoint_path, map_location=device)

# Support different key names in checkpoint
if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    # If checkpoint itself is state_dict, use it directly
    state_dict = checkpoint

# Load model to device and set evaluation mode
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# ============================================================================
# Inference on Test Set
# ============================================================================

# Lists to store all predictions and actual labels
all_preds = []
all_labels = []

# Perform inference on each batch
print("Performing inference on test set...")
with torch.inference_mode(True):
    for images, labels, _ in test_ds:
        # Move data to device
        images, labels = images.to(device), labels.to(device)
        
        # Perform inference (model returns logits, auxiliary loss and expert assignment)
        logits = model(images)
        
        # Compute probabilities and predict labels
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        # Store results
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

print(f"Inference completed. Total {len(all_labels)} samples processed.")

# ============================================================================
# Generate Classification Report
# ============================================================================
acc = accuracy_score(all_labels, all_preds)
print('Accuracy: ', acc)
# Get class names from dataset
target_names = [test_dataset.idx_to_class[i] for i in range(len(test_dataset.idx_to_class))]
print("\nDisease classes:")
print(target_names)

# Print detailed classification report
print("\n" + "="*80)
print("CLASSIFICATION REPORT")
print("="*80)
print(classification_report(all_labels, all_preds, target_names=target_names))

# ============================================================================
# Initialize Report Directory
# ============================================================================

# Create report directory if it doesn't exist
report_dir.mkdir(parents=True, exist_ok=True)
print(f"\nSaving reports to: {report_dir}")

# ============================================================================
# Visualize Confusion Matrix
# ============================================================================

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Create confusion matrix plot
fig, ax = plt.subplots(figsize=CONFUSION_MATRIX_FIGSIZE)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_names,
    yticklabels=target_names,
    annot_kws={"size": 15},
    linewidths=0.5,
    ax=ax,
)

ax.set_xlabel("Predicted Label", fontsize=15, labelpad=10)
ax.set_ylabel("True Label", fontsize=15, labelpad=10)
# ax.set_title("Confusion Matrix - Tomato Disease Classification", fontsize=14, pad=12)
ax.tick_params(axis='x', labelsize=15, rotation=45)
ax.tick_params(axis='y', labelsize=15, rotation=0)
ax.set_xticklabels(ax.get_xticklabels(), ha='right', fontsize=15)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)

# Colorbar font
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=15)

plt.tight_layout()

plt.savefig(report_dir / "confusion_matrix.png", dpi=REPORT_DPI, bbox_inches="tight")
print(f"✓ Saved confusion matrix: {report_dir / 'confusion_matrix.png'}")
plt.show()
# ============================================================================
# Visualize Classification Report
# ============================================================================

# Create classification report as dictionary
report_dict = classification_report(
    all_labels,
    all_preds,
    target_names=target_names,
    output_dict=True
)

# Convert to DataFrame for visualization
df = pd.DataFrame(report_dict).transpose()

# Create heatmap for classification report
plt.figure(figsize=CLASSIFICATION_REPORT_FIGSIZE)
sns.heatmap(
    df.iloc[:-1, :-1],
    annot=True,
    cmap="Blues",
    fmt=".2f",
    cbar_kws={"label": "Score"}
)

plt.title("Classification Report (Precision / Recall / F1-score)", fontsize=12)
plt.xlabel("Evaluation Metrics")
plt.ylabel("Disease Classes")
plt.tight_layout()

# Save image
plt.savefig(
    report_dir / "classification_report_heatmap.png",
    dpi=REPORT_DPI,
    bbox_inches="tight"
)
print(f"✓ Saved classification report: {report_dir / 'classification_report_heatmap.png'}")
plt.show()

print("\n" + "="*80)
print("MODEL EVALUATION COMPLETED")
print("="*80)
