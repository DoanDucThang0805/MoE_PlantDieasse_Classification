#!/bin/bash

source venv/bin/activate
cd src

models=(
  "trainning.shufflenetv2_train"
  "trainning.squeezenet_train"
  "trainning.efficientnetb0_train"
  "trainning.ghostnet_train"
)

seeds=(42 43 44 45 46)

for model in "${models[@]}"; do
  for seed in "${seeds[@]}"; do
    python -m "$model" --seed "$seed"
  done
done
