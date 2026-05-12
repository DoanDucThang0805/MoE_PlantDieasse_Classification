#!/bin/bash

source venv1/bin/activate
cd src

models=(
  "trainning.mobilenetv3_small_train"
)

seeds=(42)

for model in "${models[@]}"; do
  for seed in "${seeds[@]}"; do
    python -m "$model" --seed "$seed"
  done
done
