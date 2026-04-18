#!/bin/bash


source venv/bin/activate

cd src
clear
echo "=========================================="
echo "Running inference"
echo "=========================================="

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260418-120706 \
  --dataset-name plantdoc \
  --topk 2 \
  --num-expert 4 \
  --type-model moe_0.5temperature \
  --seed 52

echo "=========================================="
echo "All inference completed!"
echo "=========================================="