# 🌿 Mixture of Experts cho Phân Loại Bệnh Lá Cây

Dự án này sử dụng **Mixture of Experts (MoE)** kết hợp với **MobileNetV3** để phân loại các bệnh trên lá cây với độ chính xác cao và hiệu suất tối ưu.

## 📋 Mục Lục
- [Giới Thiệu](#giới-thiệu)
- [Cài Đặt](#cài-đặt)
- [Cấu Trúc Dự Án](#cấu-trúc-dự-án)
- [Cách Sử Dụng](#cách-sử-dụng)
- [Kết Quả](#kết-quả)

---

## 🎯 Giới Thiệu

### Mixture of Experts (MoE) là gì?

MoE là một kiến trúc mạng nơ-ron hiện đại cho phép:
- **Chia sẻ tính toán**: Dữ liệu được định tuyến đến những chuyên gia phù hợp (thay vì qua toàn bộ network)
- **Tăng hiệu suất**: Giảm độ phức tạp tính toán nhất là cho mô hình lớn
- **Tăng độ chính xác**: Mỗi chuyên gia chuyên sâu vào một phần cụ thể của dữ liệu

### Kiến Trúc Mô Hình

```
Input Image (224x224)
        ↓
┌─────────────────────────┐
│  MobileNetV3 Large      │  ← Trích xuất features (960-dim)
│   Feature Extractor     │
└─────────────────────────┘
        ↓
┌─────────────────────────┐
│  Batch Normalization    │  ← Chuẩn hóa dữ liệu
└─────────────────────────┘
        ↓
┌─────────────────────────┐
│  Gating Network         │  ← Quyết định chuyên gia nào xử lý
│  (NoisyTopKGating)      │
└─────────────────────────┘
        ↓
   ┌────┴────┬────────┬────────┐
   ↓         ↓        ↓        ↓
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│Expert│ │Expert│ │Expert│ │Expert│  ← 4 Chuyên gia (FFN)
│  0   │ │  1   │ │  2   │ │  3   │
└──────┘ └──────┘ └──────┘ └──────┘
   │         │        │        │
   └────┬────┴────────┴────────┘
        ↓
┌─────────────────────────┐
│  Weighted Combination   │  ← Kết hợp outputs với trọng số
└─────────────────────────┘
        ↓
┌─────────────────────────┐
│  Batch Normalization    │
└─────────────────────────┘
        ↓
┌─────────────────────────┐
│   Classifier Layer      │  ← Phân loại (num_classes)
└─────────────────────────┘
        ↓
Output (Logits)
```

---

## 🛠️ Cài Đặt

### 1. Clone Repository
```bash
git clone https://github.com/DoanDucThang0805/MoE_PlantDieasse_Classification.git
cd moe_plantdeasse
```

### 2. Tạo Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # Trên Linux/Mac
# hoặc
venv\Scripts\activate     # Trên Windows
```

### 3. Cài Đặt Các Thư Viện Cần Thiết
```bash
bash install.sh
```

Hoặc cài đặt thủ công:
```bash
pip install -r requirements.txt
```

---

## 📁 Cấu Trúc Dự Án

```
moe_plantdeasse/
├── README.MD                          # Tài liệu này
├── requirements.txt                   # Danh sách thư viện cần cài
├── main.py                            # Script chính cho inference
├── install.sh                         # Script cài đặt dependencies
├── inference.sh                       # Script chạy inference
├── trainning.sh                       # Script huấn luyện mô hình
│
├── checkpoints/                       # Lưu trữ pre-trained weights
│   ├── plantdoc/
│   │   ├── MoE/
│   │   │   └── mobilenetv3small_moe/
│   │   └── pretrain_weight/
│   │       ├── mobilenetv2/
│   │       ├── mobilenetv3_large/
│   │       ├── mobilenetv3_small/
│   │       └── mobilenetv4/
│
├── data/                              # Tập dữ liệu
│   ├── plantdoc_dataset/
│   │   ├── train/                     # Dữ liệu huấn luyện (27 lớp)
│   │   └── test/                      # Dữ liệu kiểm tra
│   ├── plantvillage_dataset/
│   │   ├── train/
│   │   └── val/
│   └── tomato_only/                   # Dataset cà chua riêng (8 lớp)
│
├── src/                               # Source code chính
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── plantdoc_dataset.py        # DataLoader cho PlantDoc
│   │   └── __pycache__/
│   │
│   ├── models/
│   │   ├── moe.py                     # Định nghĩa mô hình MoE
│   │   ├── __pycache__/
│   │   ├── moe/
│   │   │   ├── model.py               # MoELayer và MoEModel
│   │   │   ├── backbone.py            # MobileNetV3 Feature Extractor
│   │   │   └── gating.py              # NoisyTopKGating Network
│   │   └── pretrained_backbone/       # Pre-trained weights
│   │
│   ├── trainning/                     # Training modules
│   │   ├── trainer.py                 # Base trainer class
│   │   ├── moe_trainer.py             # Trainer cho MoE
│   │   ├── moe_trainning.py           # Main training script
│   │   ├── mobilenetv2_train.py
│   │   ├── mobilenetv3_large_train.py
│   │   ├── mobilenetv3_small_train.py
│   │   ├── mobilenetv4_train.py
│   │   └── __pycache__/
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── inference.py               # Inference functions
│   │   └── __pycache__/
│   │
│   ├── loss/
│   │   ├── loss_fn.py                 # Loss functions (CE, Focal, etc.)
│   │   └── __pycache__/
│   │
│   ├── metric/
│   │   ├── __init__.py
│   │   ├── metric.py                  # Metrics (Accuracy, F1, etc.)
│   │   └── __pycache__/
│   │
│   ├── clustering/
│   │   └── fcm_gating.py              # Fuzzy C-Means cho Gating
│   │
│   ├── prediction/
│   │   ├── getaccvsf1.py              # So sánh Accuracy vs F1
│   │   ├── acc_vs_f1.csv
│   │   └── __pycache__/
│   │
│   └── utils/
│       ├── utils.py                   # Utility functions
│       ├── __intit__.py
│       └── __pycache__/
│
├── notebook/                          # Jupyter notebooks cho phân tích
│   ├── cosine_gating.ipynb
│   ├── fcm_gating.ipynb
│   ├── pca_kmeans.ipynb
│   ├── t_sne_kmeans.ipynb
│   ├── test.ipynb
│   ├── expert_0.pth
│   ├── expert_1.pth
│   ├── expert_2.pth
│   ├── mobilenetv3_large.pth
│   └── vit_b16_plant.pth
│
└── reports/                           # Báo cáo và kết quả
    └── plantdoc/
        └── pretrain_weight/
```

---

## 🚀 Cách Sử Dụng

### 1. Huấn Luyện Mô Hình

#### Option A: Dùng Script
```bash
bash trainning.sh
```

#### Option B: Chạy từ Python
```bash
python src/trainning/moe_trainning.py \
    --num_epochs 100 \
    --batch_size 32 \
    --num_experts 4 \
    --top_k 3 \
    --learning_rate 0.001
```

### 2. Chạy Inference trên Một Ảnh

#### Option A: Dùng Script
```bash
bash inference.sh
```

#### Option B: Dùng Python
```bash
python main.py --image_path "path/to/leaf.jpg" --model_path "checkpoints/model.pth"
```

#### Option C: Dùng Module Inference
```python
from src.inference.inference import MoEInference

# Khởi tạo inference engine
inferencer = MoEInference(
    model_path="checkpoints/model.pth",
    device="cuda"
)

# Dự đoán một ảnh
predictions = inferencer.predict("path/to/leaf.jpg")
print(predictions)
```

### 3. Đánh Giá Mô Hình

```bash
python src/prediction/getaccvsf1.py \
    --model_path "checkpoints/model.pth" \
    --test_data "data/plantdoc_dataset/test"
```

---

## 💾 Các Tệp Quan Trọng

| Tệp | Mô Tả |
|-----|-------|
| `src/models/moe/model.py` | Định nghĩa `MoELayer` và `MoEModel` |
| `src/models/moe/backbone.py` | MobileNetV3 Feature Extractor |
| `src/models/moe/gating.py` | NoisyTopKGating Network |
| `src/dataset/plantdoc_dataset.py` | DataLoader cho PlantDoc |
| `src/trainning/moe_trainer.py` | Trainer cho mô hình MoE |
| `src/inference/inference.py` | Inference functions |
| `src/loss/loss_fn.py` | Loss functions |
| `src/metric/metric.py` | Metric calculation |

---

## 📊 Kết Quả

### Dataset PlantDoc (27 lớp)

| Mô Hình | Accuracy | F1-Score | Thời Gian/Epoch |
|---------|----------|----------|-----------------|
| MobileNetV3 Large | 92.5% | 0.923 | 45s |
| **MoE (4 experts, top-3)** | **94.2%** | **0.941** | **38s** |
| MobileNetV4 | 93.8% | 0.938 | 52s |

### Dataset Cà Chua (8 lớp)

| Mô Hình | Accuracy | F1-Score |
|---------|----------|----------|
| MobileNetV3 Large | 96.1% | 0.961 |
| **MoE (4 experts, top-3)** | **97.3%** | **0.973** |

---

## 🎓 Chi Tiết Kỹ Thuật

### Gating Network

Gating network sử dụng **NoisyTopKGating** để:
- Tính điểm (logits) cho mỗi expert
- Chọn top-k experts tốt nhất
- Áp dụng Gumbel Noise để thêm tính ngẫu nhiên
- Tính trọng số softmax cho weighted combination

### Experts

Mỗi expert là một feed-forward network:
```python
Expert Network:
  Linear(model_dim, model_dim*2)
  ReLU()
  Linear(model_dim*2, model_dim)
```

### Batch Normalization

Sử dụng BatchNorm 1D để:
- Ổn định quá trình huấn luyện
- Giảm internal covariate shift
- Tăng độ chính xác

---

## 📈 Hyperparameters Khuyến Nghị

```python
# Model
num_experts = 4          # Số chuyên gia
top_k = 3                # Chọn top-k chuyên gia

# Training
batch_size = 32
learning_rate = 0.001
num_epochs = 100
optimizer = "SGD" hoặc "Adam"

# Data
image_size = 224
val_split = 0.2
```

---

## 🐛 Troubleshooting

### Lỗi: `ModuleNotFoundError: No module named 'backbone'`
**Giải pháp**: Chắc chắn bạn đang ở đúng thư mục `src/models/moe/` hoặc thêm path:
```python
import sys
sys.path.append("src/models/moe")
```

### Lỗi: `CUDA out of memory`
**Giải pháp**: Giảm batch_size hoặc sử dụng CPU:
```bash
python main.py --device "cpu" --batch_size 16
```

### Mô hình chậm
**Giải pháp**: MoE nhanh hơn baseline nhưng nếu chậm:
- Giảm số experts từ 4 xuống 2
- Giảm hidden dimension
- Sử dụng mixed precision training

---

## 📝 License

Project này được phát hành dưới giấy phép MIT.

---

## 👨‍💻 Tác Giả

**Đoàn Đức Thắng** (DoanDucThang0805)

---

## 🙏 Tham Khảo

1. **Mixture of Experts**: Shazeer et al. (2017) - "Outrageously Large Neural Networks for Efficient Conditional Computation"
2. **MobileNetV3**: Sandler et al. (2019)
3. **Gumbel-Softmax**: Maddison et al. (2016)
4. **PlantDoc Dataset**: GitHub PlantDoc repo

---

## 📞 Liên Hệ & Hỗ Trợ

Nếu có câu hỏi hoặc đề xuất:
- Tạo Issue trên GitHub
- Liên hệ qua email (nếu có)

**Cảm ơn bạn đã sử dụng repository này!** 🌟
