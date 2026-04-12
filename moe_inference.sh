#!/bin/bash
# Inference script for MoE models (2 Experts, Top K = 2)
source venv/bin/activate

cd src
clear
echo "=========================================="
echo "Running inference for 2 Experts, Top K = 2"
echo "=========================================="

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260411-110211 \
  --dataset-name plantdoc \
  --topk 2 \
  --num-expert 5 \
  --type-model MoE_classweight
echo "=========================================="
echo "All inference completed!"
echo "=========================================="