#!/bin/bash
# Inference script for MoE models (2 Experts, Top K = 2)
source venv/bin/activate

cd src

# ============================================================================
# Inference: 2 Experts with Top K = 2
# ============================================================================

echo "=========================================="
echo "Running inference for 2 Experts, Top K = 2"
echo "=========================================="

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260407-142059 \
  --dataset-name plantdoc \
  --topk 2 \
  --num-expert 2

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260407-153446 \
  --dataset-name plantdoc \
  --topk 2 \
  --num-expert 2

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260407-154538 \
  --dataset-name plantdoc \
  --topk 2 \
  --num-expert 2

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260407-160154 \
  --dataset-name plantdoc \
  --topk 2 \
  --num-expert 2


echo "=========================================="
echo "All inference completed!"
echo "=========================================="