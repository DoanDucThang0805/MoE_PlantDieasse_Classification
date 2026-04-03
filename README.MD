# MoE Plant Disease Classification

## Giới Thiệu (Introduction)

**MoE Plant Disease Classification** là một dự án học sâu (Deep Learning) sử dụng **Mixture of Experts (MoE)** để phân loại bệnh lá cây trên dataset **PlantDoc**. Mô hình kết hợp kiến trúc MoE với các backbone như MobileNetV3 để đạt hiệu suất cao trong phân loại bệnh thực vật.

### Đặc Điểm Chính
- **Kiến trúc MoE (Mixture of Experts)**: Sử dụng nhiều chuyên gia (experts) với gating mechanism thông minh
- **Router Modes**: Hỗ trợ hai chế độ routing:
  - `context_aware`: Sử dụng context features để chọn experts
  - `noisy`: Sử dụng Noisy Top-K routing
- **Context Features**: Tích hợp context features từ dữ liệu hình ảnh
- **Multiple Backbones**: MobileNetV3 (Large/Small), ResNet50, EfficientNet, ViT, ShuffleNet
- **Production Ready**: Có logging, checkpoint management, và evaluation metrics đầy đủ

---

## Cấu Trúc Dự Án (Project Structure)

```
MoE_PlantDieasse_Classification/
│
├── README.MD                          # Documentation chính
├── requirements.txt                   # Python dependencies
├── main.py                            # Entry point chính
├── install.sh                         # Setup script
├── trainning.sh                       # Training execution script
├── moe_inference.sh                   # MoE inference script
├── pretrained_inference.sh            # Pretrained model inference script
│
├── data/
│   └── tomato_only/                   # PlantDoc dataset
│       ├── Tomato Early blight leaf/
│       ├── Tomato leaf/
│       ├── Tomato leaf bacterial spot/
│       ├── Tomato leaf late blight/
│       └── ... (8 disease classes)
│
├── notebook/                          # Jupyter notebooks
│   ├── cosine_gating.ipynb
│   ├── fcm_gating.ipynb
│   ├── pca_kmeans.ipynb
│   ├── t_sne_kmeans.ipynb
│   ├── test.ipynb
│   └── mobilenetv3_large.pth
│
├── reports/                           # Training & evaluation reports
│   └── plantdoc/
│       ├── MoE/
│       │   ├── mobilenetv3large_moe/
│       │   └── mobilenetv3small_moe/
│       └── pretrain_weight/
│           ├── mobilenetv3_large/
│           ├── mobilenetv3_small/
│           ├── resnet50/
│           └── shufflenet/
│
└── src/                               # Source code
    ├── dataset/
    │   ├── __init__.py
    │   └── plantdoc_dataset.py        # Dataset loading & preprocessing
    │
    ├── inference/
    │   ├── moe/
    │   │   ├── __init__.py
    │   │   └── inference.py           # MoE inference module
    │   └── pretrained/
    │       └── inference.py           # Pretrained model inference
    │
    ├── loss/
    │   └── loss_fn.py                 # Custom loss functions
    │
    ├── metric/
    │   ├── __init__.py
    │   └── metric.py                  # Evaluation metrics
    │
    ├── models/
    │   ├── moe/
    │   │   ├── __init__.py
    │   │   ├── backbone.py            # MoE backbone architectures
    │   │   ├── gating.py              # Gating mechanisms
    │   │   └── model.py               # MoE model definition
    │   ├── pretrained_backbone/       # Pretrained backbones
    │   │   ├── mobilenetv2.py
    │   │   ├── mobilenetv3_large.py
    │   │   ├── mobilenetv3_small.py
    │   │   └── mobilenetv4.py
    │   └── pretrained_model/          # Pretrained models
    │       ├── efficientnet_b4.py
    │       ├── mobilenetv3_large.py
    │       ├── mobilenetv3_small.py
    │       ├── resnet50.py
    │       ├── shufflenet.py
    │       └── vit.py
    │
    ├── trainning/
    │   ├── __init__.py
    │   ├── context_moe_train.py       # MoE training with context
    │   ├── moe_train.py               # Standard MoE training
    │   ├── mobilenetv3_large_train.py
    │   ├── mobilenetv3_small_train.py
    │   └── ... (other training scripts)
    │
    ├── utils/
    │   ├── __init__.py
    │   ├── context_features.py        # Context feature extraction
    │   ├── context_moe_trainer.py     # MoE trainer with context
    │   ├── load_dataset.py            # Dataset utilities
    │   ├── moe_trainer.py             # Standard MoE trainer
    │   └── trainer.py                 # Base trainer class
    │
    └── prediction/
        ├── __init__.py
        ├── getaccvsf1.py              # Accuracy vs F1 analysis
        └── acc_vs_f1.csv
```

---

## Installation & Setup

### 1. Clone Repository
```bash
cd f:\thang\MoE_PlantDieasse_Classification
```

### 2. Create Virtual Environment
```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Linux/MacOS:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

Or run the setup script:
```bash
bash install.sh
```

---

## Cách Sử Dụng (Usage Guide)

### Training

#### 1. **MoE Model Training with Context Features**

```bash
# Basic training with defaults
bash trainning.sh

# Custom parameters
bash trainning.sh \
  --batch_size 64 \
  --epochs 200 \
  --num_experts 6 \
  --top_k 2 \
  --model_name mobilenetv3small_moe \
  --router_mode noisy \
  --use_context False
```

**CLI Parameters:**
- `--batch_size` (int): Batch size for training (default: 32)
- `--epochs` (int): Number of training epochs (default: 300)
- `--num_experts` (int): Number of experts in MoE (default: 8)
- `--top_k` (int): Top-K experts to select (default: 4)
- `--model_name` (str): Model architecture name (default: mobilenetv3large_moe)
- `--type_model` (str): Model type (default: MoE)
- `--router_mode` (str): context_aware or noisy (default: context_aware)
- `--use_context` (bool): Use context features - True/False (default: True)

#### 2. **Direct Python Training**

```bash
# From project root
cd src
python -m trainning.context_moe_train \
  --batch_size 32 \
  --epochs 300 \
  --num_experts 8 \
  --top_k 4 \
  --model_name mobilenetv3large_moe \
  --type_model MoE \
  --router_mode context_aware \
  --use_context True
```

#### 3. **Training Configuration**

Config parameters được lưu trong `Config` class và có thể điều chỉnh:

```python
# src/trainning/context_moe_train.py
Config.batch_size = 64          # Batch size
Config.epochs = 300             # Training epochs
Config.num_experts = 8          # Number of experts
Config.top_k = 4                # Top-K selection
Config.model_name = 'mobilenetv3large_moe'
Config.router_mode = 'context_aware'
```

---

### Inference

#### 1. **MoE Model Inference (Context-Aware)**

```bash
# Basic inference (extracts hyperparameters from checkpoint)
python -m inference.moe.context_aware_moe_inference \
  --run_time run_20260320-155951

# Full parameters
python -m inference.moe.context_aware_moe_inference \
  --model_name mobilenetv3large_moe \
  --type_model MoE \
  --run_time run_20260320-155951 \
  --dataset_name plantdoc \
  --use_context True \
  --router_mode context_aware

# Without context, noisy router
python -m inference.moe.context_aware_moe_inference \
  --run_time run_20260320-155951 \
  --use_context False \
  --router_mode noisy
```

**CLI Parameters:**
- `--model_name` (str): Model architecture name (optional)
- `--type_model` (str): Model type (optional)
- `--run_time` (str): Training run timestamp - **REQUIRED** (e.g., run_20260320-155951)
- `--dataset_name` (str): Dataset name (optional, default: plantdoc)
- `--use_context` (bool): Use context features - True/False (default: True)
- `--router_mode` (str): context_aware or noisy (default: context_aware)

#### 2. **Bash Inference Script**

```bash
bash moe_inference.sh
```

#### 3. **Inference Output**

Reports được lưu tại:
```
reports/plantdoc/MoE/{model_name}/{run_time}/
├── confusion_matrix.png               # Confusion matrix visualization
├── classification_report_heatmap.png  # Precision/Recall/F1 heatmap
└── [model_name]_report.txt           # Detailed classification report
```

---

## Model Architecture Details

### MoE Components

#### 1. **Backbone** (Feature Extraction)
- MobileNetV3 (Large/Small)
- ResNet50
- EfficientNet-B4
- ViT (Vision Transformer)
- ShuffleNet

#### 2. **Experts**
- Multiple expert networks với số lượng tuỳ chỉnh
- Mỗi expert là một fully connected layer
- Outputs từ tất cả experts được combine

#### 3. **Gating Mechanisms**
- **Context-Aware**: Sử dụng context features để tính gating weights
- **Noisy Top-K**: Thêm noise và chọn Top-K experts

#### 4. **Context Features** (6 dimensions)
- Image statistics (mean, std, min, max)
- Domain-specific features
- Extracted từ input images

### Model Hyperparameters

Tất cả hyperparameters được lưu trong checkpoint và tự động tích hợp:

```python
num_classes: int = 8          # Number of disease classes
num_experts: int = 8          # Number of experts
context_dim: int = 6          # Context feature dimension
top_k: int = 4                # Top-K experts selection
router_mode: str = 'context_aware'  # Routing mechanism
```

---

## Training Configuration Details

### Loss Functions
```python
MoELoss:
  - Primary loss: Cross-entropy loss
  - Auxiliary loss: Expert load balancing (alpha=0.05)
```

### Optimizer
```python
Adam:
  - Learning rate: 0.001
  - Weight decay: 0.001
```

### Data Batch Components
```python
# With context features
(images, labels, context_features) = batch
shape: (batch_size, 3, 224, 224), (batch_size,), (batch_size, 6)

# Without context
(images, labels) = batch
shape: (batch_size, 3, 224, 224), (batch_size,)
```

---

## Dataset Information

### PlantDoc Dataset
- **Classes**: 8 tomato disease types
- **Folder**: `data/tomato_only/`
- **Classes**:
  1. Tomato Early blight leaf
  2. Tomato leaf
  3. Tomato leaf bacterial spot
  4. Tomato leaf late blight
  5. Tomato leaf mosaic virus
  6. Tomato leaf yellow virus
  7. Tomato mold leaf
  8. Tomato Septoria leaf spot

### Data Loading
```bash
# Automatic dataset loading with context features
from src.dataset.plantdoc_dataset import build_datasets

train_loader, val_loader, test_loader = build_datasets(
    use_context=True,
    batch_size=32
)
```

---

## Checkpoint Management

### Checkpoint Structure
```
reports/plantdoc/MoE/{model_name}/{run_time}/
└── best_checkpoint.pth
```

### Checkpoint Contents
```python
{
    'model_state_dict': {...},      # Model weights
    'epoch': int,                    # Training epoch
    'best_val_acc': float,          # Best validation accuracy
    'num_classes': int,             # Number of classes
    'num_experts': int,             # Number of experts
    'context_dim': int,             # Context dimension
    'top_k': int,                   # Top-K value
    'router_mode': str,             # Router mode used
}
```

### Loading Checkpoint
```python
checkpoint = torch.load('best_checkpoint.pth', map_location='cpu')
metadata = checkpoint  # Contains hyperparameters
model_state = checkpoint['model_state_dict']
```

---

## Evaluation Metrics

### Generated Reports
- **Confusion Matrix**: Visual comparison of predicted vs true labels
- **Classification Report**: Precision, Recall, F1-Score per class
- **Accuracy**: Overall model accuracy
- **Macro/Weighted F1**: Aggregated performance metrics

### Report Visualization
```
confusion_matrix.png          # Heatmap of model predictions
classification_report_heatmap.png  # Precision/Recall scores
```

---

## Advanced Configuration

### Custom Model Training
```bash
# For other backbones
python -m trainning.mobilenetv3_small_train --batch_size 64
python -m trainning.resnet50_train --epochs 250
python -m trainning.efficientnetb4_train --batch_size 128
```

### Router Modes Comparison
```bash
# Context-Aware (recommended)
--router_mode context_aware --use_context True

# Noisy Top-K
--router_mode noisy --use_context False
```

### Batch Size Optimization
```bash
# Smaller GPU memory
bash trainning.sh --batch_size 16

# Larger batch (requires more VRAM)
bash trainning.sh --batch_size 128
```

---

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
bash trainning.sh --batch_size 16
```

### Checkpoint Not Found
```bash
# Verify run_time matches checkpoint directory
# Check: reports/plantdoc/MoE/{model_name}/{run_time}/
python -m inference.moe.context_aware_moe_inference --run_time run_20260320-155951
```

### Context Features Not Available
```bash
# Ensure use_context=True matches training
python -m inference.moe.context_aware_moe_inference \
  --run_time {run_time} \
  --use_context True
```

---

## Dependencies

See `requirements.txt` for complete list. Key packages:
- **PyTorch**: Deep learning framework
- **torchinfo**: Model summary
- **scikit-learn**: Metrics and evaluation
- **matplotlib/seaborn**: Visualization
- **pandas/numpy**: Data processing
- **Pillow**: Image processing

---

## Author & Version
- **Project**: MoE Plant Disease Classification
- **Version**: 2.0
- **Last Updated**: April 2026

---

## License & Credits
Plant disease classification using Mixture of Experts architecture on PlantDoc dataset.
