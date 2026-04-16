#!/bin/bash


source venv/bin/activate

cd src
clear
echo "=========================================="
echo "Running inference"
echo "=========================================="

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260416-175227 \
  --dataset-name plantdoc \
  --topk 2 \
  --num-expert 4 \
  --type-model moe_0.5temperature \
  --seed 42
python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260416-171337 \
  --dataset-name plantdoc \
  --topk 2 \
  --num-expert 4 \
  --type-model moe_0.5temperature \
  --seed 42
python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260416-182638 \
  --dataset-name plantdoc \
  --topk 2 \
  --num-expert 4 \
  --type-model moe_0.5temperature \
  --seed 42
python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260416-191039 \
  --dataset-name plantdoc \
  --topk 2 \
  --num-expert 4 \
  --type-model moe_0.5temperature \
  --seed 42
echo "=========================================="
echo "All inference completed!"
echo "=========================================="