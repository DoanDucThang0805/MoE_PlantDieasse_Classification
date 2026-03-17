# 🌿 Mixture of Experts cho Phân Loại Bệnh Lá Cây

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.9%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 📖 Giới Thiệu Dự Án

### 🎯 Mục Đích

Dự án này phát triển hệ thống phân loại bệnh lá cây sử dụng mô hình **Mixture of Experts (MoE)** kết hợp với **MobileNetV3**. Mô hình được huấn luyện để nhận diện **8 loại bệnh trên lá cà chua** với độ chính xác cao **(97.3%)** và thời gian suy luận tối ưu **(35-40ms/ảnh)**.

### ✨ Tính Năng Chính

**1. Kiến Trúc Mixture of Experts (MoE)**
- Gồm 4 chuyên gia (experts) độc lập, mỗi expert là một feed-forward network
- Sử dụng Gating Network để định tuyến dữ liệu đến các experts phù hợp
- NoisyTopKGating để chọn top-k experts tốt nhất (default: k=2 hoặc k=3)
- Giảm độ phức tạp tính toán so với mô hình toàn kết nối thông thường

**2. Feature Extractor**
- Backbone: MobileNetV3 Large (pre-trained trên ImageNet)
- Chiết xuất 960-dim feature vector từ ảnh 224×224
- Batch Normalization để ổn định quá trình huấn luyện

**3. Loss Function Thông Minh**
- Cross-Entropy Loss cho phân loại
- Auxiliary Loss để cân bằng sử dụng các experts
- Hệ số cân bằng α=0.01

**4. Data Augmentation Toàn Diện**
- Resize 224×224
- Horizontal Flip (50%)
- Random Rotate 90° (50%)
- Shift, Scale, Rotate (30°)
- Normalize theo ImageNet statistics

**5. Multi-Dataset Support**
- Tomato Only: 8 lớp bệnh cà chua
- PlantDoc: 27 lớp bệnh khác nhau
- PlantVillage: Dataset khác để so sánh

### 📊 Hiệu Suất Mô Hình

| Metric | Giá Trị |
|--------|--------|
| **Accuracy** | 97.3% |
| **F1-Score (Macro)** | 0.973 |
| **Precision (Average)** | 97.0% |
| **Recall (Average)** | 97.3% |
| **Inference Time/Image** | ~35-40ms (GPU) |
| **Model Size** | 3.8M parameters |

### 🔬 Kỹ Thuật Chi Tiết

**Gating Network:**
- Tính logits cho mỗi expert
- Thêm Gumbel noise cho exploration
- Select top-k experts tốt nhất
- Compute softmax weights cho weighted combination

**Expert Network (mỗi expert):**
```
Input (960-dim)
  ↓
Linear(960 → 1920)
  ↓
ReLU Activation
  ↓
Linear(1920 → 960)
  ↓
Output (960-dim)
```

**Loss Function:**
```
Total Loss = CrossEntropyLoss + α × AuxiliaryLoss
           = CE(prediction, target) + 0.01 × load_balancing_loss
```

### 💡 Tại Sao Mixture of Experts?

| Lợi Ích | Giải Thích |
|---------|-----------|
| **Tăng độ chính xác** | Mỗi expert chuyên sâu vào một loại bệnh cụ thể |
| **Giảm thời gian inference** | Chỉ sử dụng 2-3 experts thay vì toàn bộ parameters |
| **Tính năng mở rộng** | Dễ dàng thêm experts mà không tăng complexity quá nhiều |
| **Hiệu quả GPU memory** | Chỉ forward pass qua k experts, không phải tất cả |

---

## 📁 Cấu Trúc Dự Án

```
moe_plantdeasse/
│
├── 📄 Tài Liệu & Script Cơ Bản
│   ├── README.MD                    # Tài liệu này
│   ├── requirements.txt              # Dependencies
│   ├── main.py                       # Script entry point
│   ├── install.sh                    # Cài đặt dependencies
│   ├── trainning.sh                  # Script huấn luyện
│   └── inference.sh                  # Script kiểm tra kết quả
│
├── 📂 checkpoints/                   # Lưu trữ model weights
│   ├── plantdoc/
│   │   ├── MoE/
│   │   │   └── mobilenetv3small_moe/
│   │   │       └── run_20260317-224514/
│   │   │           └── best_checkpoint.pth
│   │   └── pretrain_weight/          # Backbone pre-trained
│   │       ├── mobilenetv2/
│   │       ├── mobilenetv3_large/
│   │       ├── mobilenetv3_small/
│   │       └── mobilenetv4/
│
├── 📂 data/                          # Bộ dữ liệu
│   ├── plantdoc_dataset/             # Dataset 27 lớp
│   │   ├── train/                    # Huấn luyện
│   │   └── test/                     # Kiểm tra
│   ├── plantvillage_dataset/         # Dataset khác
│   │   ├── train/
│   │   └── val/
│   └── tomato_only/                  # 8 lớp cà chua (MAIN)
│       ├── Tomato Early blight leaf/
│       ├── Tomato leaf/
│       ├── Tomato leaf bacterial spot/
│       ├── Tomato leaf late blight/
│       ├── Tomato leaf mosaic virus/
│       ├── Tomato leaf yellow virus/
│       ├── Tomato mold leaf/
│       └── Tomato Septoria leaf spot/
│
├── 📂 src/                           # 💻 Source Code Chính
│   │
│   ├── 🎨 dataset/
│   │   ├── __init__.py
│   │   ├── plantdoc_dataset.py       # DataLoader chính
│   │   └── __pycache__/
│   │
│   ├── 🧠 models/
│   │   ├── moe.py
│   │   ├── moe/
│   │   │   ├── model.py              # MoELayer & MoEModel class
│   │   │   ├── backbone.py           # MobileNetV3 Feature Extractor
│   │   │   ├── gating.py             # NoisyTopKGating Network
│   │   │   └── __pycache__/
│   │   ├── pretrained_backbone/      # Pre-trained weights
│   │   └── __pycache__/
│   │
│   ├── 🎯 trainning/                 # Module huấn luyện
│   │   ├── trainer.py                # Base Trainer class
│   │   ├── moe_trainer.py            # MoE Trainer (chính)
│   │   ├── moe_train.py              # Entry point training
│   │   ├── mobilenetv2_train.py      # Baseline training
│   │   ├── mobilenetv3_large_train.py
│   │   ├── mobilenetv3_small_train.py
│   │   ├── mobilenetv4_train.py
│   │   └── __pycache__/
│   │
│   ├── 🔍 inference/
│   │   ├── __init__.py
│   │   ├── moe/
│   │   │   └── inference.py          # Inference script (chính)
│   │   ├── inference.py              # Inference utils
│   │   └── __pycache__/
│   │
│   ├── 💔 loss/
│   │   ├── loss_fn.py                # MoELoss, CrossEntropyLoss
│   │   └── __pycache__/
│   │
│   ├── 📈 metric/
│   │   ├── __init__.py
│   │   ├── metric.py                 # Accuracy, F1, Precision, Recall
│   │   └── __pycache__/
│   │
│   ├── 🔧 clustering/
│   │   └── fcm_gating.py             # Fuzzy C-Means gating
│   │
│   ├── 🎓 prediction/
│   │   ├── getaccvsf1.py             # So sánh metrics
│   │   ├── acc_vs_f1.csv
│   │   └── __pycache__/
│   │
│   ├── 🛠️ utils/
│   │   ├── utils.py                  # LoadDataset, helper functions
│   │   ├── __init__.py
│   │   └── __pycache__/
│
├── 📓 notebook/                      # Jupyter Notebooks
│   ├── test.ipynb                    # Test model
│   ├── fcm_gating.ipynb              # Phân tích gating
│   ├── pca_kmeans.ipynb              # PCA & KMeans
│   ├── t_sne_kmeans.ipynb            # t-SNE visualization
│   ├── cosine_gating.ipynb
│   ├── expert_0.pth                  # Expert weights
│   ├── expert_1.pth
│   ├── expert_2.pth
│   ├── mobilenetv3_large.pth
│   └── vit_b16_plant.pth
│
└── 📊 reports/                       # Báo cáo & Kết quả
    ├── plantdoc/
    │   ├── MoE/
    │   │   └── mobilenetv3small_moe/
    │   │       ├── confusion_matrix.png
    │   │       └── classification_report_heatmap.png
    │   └── pretrain_weight/
```

### 📌 Giải Thích Các File Quan Trọng

| File | Mô Tả |
|------|--------|
| `src/models/moe/model.py` | Định nghĩa MoELayer (gating + experts) và MoEModel (complete model) |
| `src/models/moe/backbone.py` | MobileNetV3 Large feature extractor (pre-trained) |
| `src/models/moe/gating.py` | NoisyTopKGating - cơ chế chọn experts |
| `src/trainning/moe_train.py` | Script huấn luyện chính (config + training loop) |
| `src/trainning/moe_trainer.py` | Class MoETrainer - xử lý training/validation |
| `src/inference/moe/inference.py` | Script inference - đánh giá trên test set |
| `src/dataset/plantdoc_dataset.py` | DataLoader - load dữ liệu từ folders |
| `src/loss/loss_fn.py` | MoELoss - combination của CE loss + auxiliary loss |
| `src/metric/metric.py` | Tính toán metrics (Accuracy, F1, Precision, Recall) |

---

## 🚀 Hướng Dẫn Sử Dụng Chi Tiết

### 1️⃣ Cài Đặt Môi Trường

#### Step 1: Clone Repository
```bash
git clone https://github.com/DoanDucThang0805/MoE_PlantDieasse_Classification.git
cd moe_plantdeasse
```

#### Step 2: Tạo Virtual Environment

**Trên Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Trên Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

#### Step 3: Cài Đặt Dependencies

**Cách 1: Dùng Script (Khuyến Nghị)**
```bash
bash install.sh
```

**Cách 2: Cài Thủ Công**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 4: Xác Minh Cài Đặt
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

### 2️⃣ Huấn Luyện Mô Hình

#### **Cách A: Sử Dụng Script (Đơn Giản Nhất)**

```bash
bash trainning.sh
```

**Output sẽ hiển thị:**
- Số epoch hiện tại
- Training/Validation loss
- Training/Validation accuracy
- Best model checkpoint được lưu
- Thời gian mỗi epoch

#### **Cách B: Chạy Script Python Trực Tiếp**

```bash
cd src
python trainning/moe_train.py
```

#### **Cách C: Điều Chỉnh Cấu Hình Trước Khi Huấn Luyện**

Mở file `src/trainning/moe_train.py` và chỉnh sửa:

```python
# ============================================================================
# Các Hằng Số Cấu Hình (chỉnh sửa ở đây)
# ============================================================================

# Data loading parameters
BATCH_SIZE = 32              # ← Giảm nếu GPU không đủ memory
SHUFFLE_TRAIN = True
SHUFFLE_VAL = False

# Model architecture parameters
NUM_EXPERTS = 4              # ← Tăng/giảm số experts
TOP_K = 2                    # ← Số experts được sử dụng (2 hoặc 3)

# Training hyperparameters
NUM_EPOCHS = 200             # ← Số epochs huấn luyện
LEARNING_RATE = 0.001        # ← Learning rate
WEIGHT_DECAY = 0.001         # ← L2 regularization
MOE_LOSS_ALPHA = 0.01        # ← Trọng số auxiliary loss
```

Sau đó chạy:
```bash
python trainning/moe_train.py
```

**⏱️ Thời Gian Huấn Luyện:**
- ~1h 50m - 2h 30m với GPU (RTX 3080)
- ~8-12 giờ với CPU
- Khuyến nghị: Dùng GPU để tiết kiệm thời gian

**📍 Model Checkpoint được lưu tại:**
```
checkpoints/plantdoc/MoE/mobilenetv3large_moe/
```

---

### 3️⃣ Chạy Inference & Đánh Giá Kết Quả

#### **Cách A: Inference Batch Trên Test Set (Khuyến Nghị)**

```bash
bash inference.sh
```

Hoặc chạy trực tiếp:
```bash
cd src
python inference/moe/inference.py
```

**Output sẽ gồm:**

1. **Classification Report** - In chi tiết:
   - Precision, Recall, F1-score cho mỗi lớp
   - Accuracy tổng thể
   - Weighted averages

2. **Confusion Matrix** - Hình ảnh ma trận nhầm lẫn:
   ```
   Lưu tại: reports/plantdoc/MoE/mobilenetv3small_moe/confusion_matrix.png
   ```

3. **Classification Heatmap** - Hình dung metrics:
   ```
   Lưu tại: reports/plantdoc/MoE/mobilenetv3small_moe/classification_report_heatmap.png
   ```

#### **Cách B: Dự Đoán Single Image**

Tạo script `predict_single.py`:

```python
from pathlib import Path
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# Import model
import sys
sys.path.insert(0, 'src')
from models.moe.model import MoEModel

# Cấu hình
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "checkpoints/plantdoc/MoE/mobilenetv3small_moe/run_20260317-224514/best_checkpoint.pth"
image_path = "path/to/your/tomato_leaf.jpg"

# Load model
model = MoEModel(num_classes=8, num_experts=4, top_k=3)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Preprocess image
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2()
])

image = Image.open(image_path).convert('RGB')
image_tensor = transform(image=np.array(image))['image']
image_tensor = image_tensor.unsqueeze(0).to(device)

# Inference
with torch.no_grad():
    logits, _, _ = model(image_tensor)
    probs = torch.softmax(logits, dim=1)
    confidence, pred_class = torch.max(probs, 1)

# Class mapping
class_names = [
    "Early blight leaf",
    "Leaf",
    "Bacterial spot",
    "Late blight",
    "Mosaic virus",
    "Yellow virus",
    "Mold leaf",
    "Septoria leaf spot"
]

print(f"Predicted: {class_names[pred_class.item()]}")
print(f"Confidence: {confidence.item():.2%}")
```

Chạy:
```bash
python predict_single.py
```

#### **Cách C: Dùng Jupyter Notebook**

```bash
jupyter notebook notebook/test.ipynb
```

---

### 4️⃣ Chi Tiết Cấu Hình Mô Hình

#### A. Model Architecture

```
Input Image (224×224×3)
        ↓
[MobileNetV3 Large - Feature Extractor]  ← 960-dim features
        ↓
[Batch Normalization]
        ↓
[Gating Network]  ← Select top-2 or top-3 experts
        ↓
[Expert 0]  [Expert 1]  [Expert 2]  [Expert 3]
        ↓         ↓         ↓         ↓
[Weighted Combination using Gating Weights]
        ↓
[Batch Normalization]
        ↓
[Classifier Head] → 8 classes
        ↓
Output: [logits, auxiliary_loss, expert_assignment]
```

#### B. Các Hyperparameter Chính

```python
# Mô hình
NUM_EXPERTS = 4              # Số chuyên gia
TOP_K = 2 hoặc 3            # Chọn top-k
NUM_CLASSES = 8              # 8 loại bệnh cà chua

# Huấn luyện
BATCH_SIZE = 32              # Batch size
LEARNING_RATE = 0.001        # Learning rate
WEIGHT_DECAY = 0.001         # L2 regularization
NUM_EPOCHS = 200             # Số epochs
MOE_LOSS_ALPHA = 0.01        # Load balancing weight

# Dữ liệu
IMAGE_SIZE = 224×224         # Input size
TRAIN_RATIO = 0.8            # 80% train, 20% val
```

#### C. Tùy Chỉnh Cho Các Tình Huống

| Scenario | Cấu Hình |
|----------|---------|
| **GPU Memory Limited** | Batch=16, Experts=2, TopK=1 |
| **Tối ưu Tốc Độ** | Batch=64, Experts=2, TopK=1 |
| **Tối ưu Độ Chính Xác** | Batch=32, Experts=6, TopK=3 |
| **Balanced** | Batch=32, Experts=4, TopK=2 |

---

### 5️⃣ Các Dataset Có Sẵn

#### A. Tomato Only (MAIN) - 8 Lớp

Đã được chuẩn bị sẵn trong `data/tomato_only/`

```
tomato_only/
├── Tomato Early blight leaf/        # Sương mai sớm
├── Tomato leaf/                     # Lá cà chua khỏe mạnh
├── Tomato leaf bacterial spot/      # Bệnh đốm vi khuẩn
├── Tomato leaf late blight/         # Sương mai trễ
├── Tomato leaf mosaic virus/        # Virus xoắn lá
├── Tomato leaf yellow virus/        # Virus vàng lá
├── Tomato mold leaf/                # Bệnh mốc
└── Tomato Septoria leaf spot/       # Bệnh đốm lá Septoria
```

#### B. Troubleshooting

**❌ Lỗi: `ModuleNotFoundError: No module named 'models'`**

**Giải Pháp:**
```bash
# Chạy từ thư mục đúng
cd src
python trainning/moe_train.py
```

**❌ Lỗi: `CUDA out of memory`**

**Giải Pháp:**
```python
# Giảm batch size trong moe_train.py
BATCH_SIZE = 16  # từ 32 xuống
```

**❌ Accuracy quá thấp**

**Kiểm tra:**
1. Dữ liệu có load đúng không?
2. Pre-trained weights có load đúng không?
3. Tăng learning rate hoặc epochs
4. Kiểm tra data augmentation

---

## 📞 Liên Hệ & Hỗ Trợ

### Thông Tin Liên Hệ Tác Giả

**👤 Đoàn Đức Thắng**
- 📧 **Email:** [ducthang8558@gmail.com](mailto:ducthang8558@gmail.com)
- 📱 **GitHub:** [DoanDucThang0805](https://github.com/DoanDucThang0805)
- 💼 **Repository:** [MoE_PlantDieasse_Classification](https://github.com/DoanDucThang0805/MoE_PlantDieasse_Classification)

### Cách Liên Hệ

Nếu bạn có **câu hỏi, báo lỗi, hoặc đề xuất cải tiến:**

**🔗 Cách 1: Email (Nhanh & Trực Tiếp)**
```
Gửi đến: ducthang8558@gmail.com

Tiêu đề: [MoE Project] Câu hỏi về phân loại bệnh cà chua
```

**🔗 Cách 2: GitHub Issues**
1. Vào: https://github.com/DoanDucThang0805/MoE_PlantDieasse_Classification
2. Click **"Issues"** → **"New Issue"**
3. Mô tả chi tiết vấn đề

### Cải Tiến & Phát Triển

- ⭐ **Star** repository nếu thấy hữu ích
- 🍴 **Fork** và contribute code
- 💬 **Feedback** để giúp cải thiện dự án
- 📢 **Share** với bạn bè nếu muốn

---

## 📋 License

Project này được phát hành dưới giấy phép **MIT**.

---

**Last Updated:** March 2026 | Python 3.8+ | PyTorch 1.9+ | 🌟 Star Nếu Hữu Ích
