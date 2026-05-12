#!/bin/bash

source venv/bin/activate
cd src

models=(
  "trainning.mobilenetv3_small_train"
)

seeds=(42 43 44 45 46)

for model in "${models[@]}"; do
  for seed in "${seeds[@]}"; do
    python -m "$model" --seed "$seed"
  done
done
