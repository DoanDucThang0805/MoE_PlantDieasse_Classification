# Hướng dẫn chạy Static INT8 Quantization & Evaluation

## Tổng quan

Pipeline gồm 3 bước:

1. **Quantize**: Export FP32 → Quantize INT8 cho mỗi pretrained model
2. **Edge Benchmark**: Đo model size, inference time, peak memory (FP32 vs INT8)
3. **Accuracy Eval**: Đo accuracy & macro-F1 drop trên PlantDoc dataset

---

## Cấu trúc file

```
├── quantize_pretrain.sh                           # Bước 1: chạy quantize tất cả models
├── eval_int8.sh                                   # Bước 2+3: chạy benchmark + eval
│
├── src/quantization/
│   ├── quantize_utils.py                          # Shared utilities
│   ├── quantize_efficientnetb0.py                 # Quantize EfficientNet-B0
│   ├── quantize_ghostnet.py                       # Quantize GhostNet
│   ├── quantize_mobilenetv3_small.py              # Quantize MobileNetV3-Small
│   ├── quantize_mobilevits.py                     # Quantize MobileViT-S
│   ├── quantize_mobilevitxs.py                    # Quantize MobileViT-XS
│   ├── quantize_shufflenetv2.py                   # Quantize ShuffleNet-V2
│   ├── quantize_squeezenet.py                     # Quantize SqueezeNet
│   ├── edge_benchmark_int8.py                     # Edge benchmark FP32 vs INT8
│   └── eval_int8_accuracy.py                      # Accuracy & mF1 evaluation
│
├── onnx/quantized/                                # Output ONNX files
│   ├── {model}_fp32.onnx
│   └── {model}_int8.onnx
│
└── results/                                       # Output CSV
    ├── edge_benchmark_int8_results.csv
    └── eval_int8_plantdoc.csv
```

---

## Bước 1: Quantize Models

### Cách 1: Chạy tất cả 7 models (dùng shell script)

Mở file `quantize_pretrain.sh`, điền checkpoint path cho từng model:

```bash
# Mở file và fill checkpoint paths
nano quantize_pretrain.sh
```

```bash
# Ví dụ:
CKPT_GHOSTNET="checkpoints/slif_tomato_dataset_phase1/pretrain_models/ghostnet/seed_42/run_20260617-150823/best_checkpoint.pth"
CKPT_MOBILENETV3_SMALL="checkpoints/slif_tomato_dataset_phase1/pretrain_models/mobilenetv3_small/seed_42/run_20260617-013532/best_checkpoint.pth"
CKPT_MOBILEVITS="checkpoints/slif_tomato_dataset_phase1/pretrain_models/mobilevits/seed_42/run_20260618-000044/best_checkpoint.pth"
CKPT_SHUFFLENETV2="checkpoints/slif_tomato_dataset_phase1/pretrain_models/shufflenetv2/seed_42/run_20260617-040215/best_checkpoint.pth"
# Để trống nếu chưa có checkpoint
CKPT_EFFICIENTNETB0=""
CKPT_MOBILEVITXS=""
CKPT_SQUEEZENET=""
```

Chạy:

```bash
bash quantize_pretrain.sh
```

### Cách 2: Chạy từng model riêng lẻ

```bash
source venv1/bin/activate

# Ví dụ: GhostNet
python src/quantization/quantize_ghostnet.py \
    --checkpoint "checkpoints/slif_tomato_dataset_phase1/pretrain_models/ghostnet/seed_42/run_20260617-150823/best_checkpoint.pth" \
    --output_dir onnx/quantized \
    --calib_size 200

# Ví dụ: MobileNetV3-Small
python src/quantization/quantize_mobilenetv3_small.py \
    --checkpoint "checkpoints/slif_tomato_dataset_phase1/pretrain_models/mobilenetv3_small/seed_42/run_20260617-013532/best_checkpoint.pth" \
    --output_dir onnx/quantized \
    --calib_size 200
```

### Tham số

| Tham số | Mô tả | Mặc định |
|---------|--------|----------|
| `--checkpoint` | Path đến `best_checkpoint.pth` | None (dùng pretrained ImageNet) |
| `--output_dir` | Thư mục lưu ONNX files | `onnx/quantized` |
| `--calib_size` | Số samples calibration từ train set | `200` |
| `--num_classes` | Số lớp phân loại | `8` |

### Output mỗi model

```
============================================================
  Static INT8 Quantization Results: ghostnet
============================================================
Model                |   Accuracy |   macro-F1 |  Size (MB)
------------------------------------------------------------
FP32 (original)      |     0.9663 |     0.9653 |      15.50
INT8 (quantized)     |     0.9476 |     0.9414 |       4.60
------------------------------------------------------------
mF1 Drop             |            |    -0.0238 |     -70.4%
============================================================
```

---

## Bước 2+3: Edge Benchmark + Accuracy Evaluation

### Chạy cả 2 cùng lúc

```bash
bash eval_int8.sh
```

### Chạy riêng Edge Benchmark

Đo model size, inference time, peak memory cho FP32 vs INT8:

```bash
source venv1/bin/activate

python src/quantization/edge_benchmark_int8.py \
    --output_dir onnx/quantized \
    --csv_store_dir results \
    --csv_filename edge_benchmark_int8_results.csv \
    --export_csv
```

### Chạy riêng Accuracy Evaluation

Đo accuracy & macro-F1 drop trên dataset:

```bash
source venv1/bin/activate

# Trên PlantDoc dataset
python src/quantization/eval_int8_accuracy.py \
    --output_dir onnx/quantized \
    --dataset plantdoc \
    --csv_store_dir results \
    --csv_filename eval_int8_plantdoc.csv \
    --export_csv

# Trên SLIF Tomato dataset
python src/quantization/eval_int8_accuracy.py \
    --output_dir onnx/quantized \
    --dataset slif_tomato \
    --csv_store_dir results \
    --csv_filename eval_int8_slif_tomato.csv \
    --export_csv
```

---

## Chạy toàn bộ pipeline (3 bước)

```bash
# Bước 1: Fill checkpoint paths rồi chạy quantize
nano quantize_pretrain.sh
bash quantize_pretrain.sh

# Bước 2+3: Edge benchmark + accuracy evaluation
bash eval_int8.sh
```

---

## Deploy trên Raspberry Pi 5

Copy file INT8 sang Pi và chạy inference:

```bash
# Trên Pi 5
pip install onnxruntime

# Inference
python -c "
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('ghostnet_int8.onnx', providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
output = session.run(None, {input_name: dummy})
print('Prediction:', np.argmax(output[0]))
"
```
