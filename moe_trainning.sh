#!/bin/bash

# load conda
source venv/bin/activate

cd src
clear

python -m trainning.context_moe_train \
    --batch_size 32 \
    --epochs 300 \
    --num_experts 5 \
    --top_k 2 \
    --model_name mobilenetv3small_moe \
    --type_model noisy_moe \
    --router_mode noisy \
    --temperature 1.0 \
    --no_context \
    --dataset_name mixed_dataset
