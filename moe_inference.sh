#!/bin/bash


source venv/bin/activate

cd src
clear
echo "=========================================="
echo "Running inference"
echo "=========================================="

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260412-160233 \
  --dataset-name plantdoc \
  --topk 2 \
  --num-expert 4 \
  --type-model MoE \
  --seed 42 
echo "=========================================="
echo "All inference completed!"
echo "=========================================="