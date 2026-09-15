#!/bin/bash
# =============================================================================
# Edge Benchmark + Accuracy Evaluation for INT8 Quantized Models
#
# 1. Edge benchmark: model size, inference time, peak memory (FP32 vs INT8)
# 2. Accuracy evaluation: accuracy & macro-F1 drop on PlantDoc dataset
#
# Prerequisite: Run quantize_pretrain.sh first to generate *_fp32.onnx
#               and *_int8.onnx files in onnx/quantized/
# =============================================================================

set -e

cd /media/data/minhht/context_moe
source venv1/bin/activate

OUTPUT_DIR="onnx/quantized"

echo "============================================================"
echo "  INT8 Quantization Evaluation Pipeline"
echo "============================================================"

# ---- 1. Edge Benchmark: size, latency, memory ----
echo ""
echo "[1/2] Running Edge Benchmark (FP32 vs INT8) ..."
python src/quantization/edge_benchmark_int8.py \
    --output_dir "${OUTPUT_DIR}" \
    --csv_store_dir "results" \
    --csv_filename "edge_benchmark_int8_results.csv" \
    --export_csv

# ---- 2. Accuracy & mF1 drop on PlantDoc ----
echo ""
echo "[2/2] Evaluating Accuracy & mF1 on PlantDoc ..."
python src/quantization/eval_int8_accuracy.py \
    --output_dir "${OUTPUT_DIR}" \
    --dataset plantdoc \
    --csv_store_dir "results" \
    --csv_filename "eval_int8_plantdoc.csv" \
    --export_csv

echo ""
echo "============================================================"
echo "  Evaluation complete!"
echo "  Results saved in: results/"
echo "============================================================"
