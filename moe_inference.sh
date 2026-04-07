#!/bin/bash
# Installation script for the application
source venv/bin/activate

cd src
python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260407-115329 \
  --dataset-name plantdoc \
  --topk 1 \
  --num-expert 2