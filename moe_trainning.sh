#!/bin/bash

# load conda
source venv/bin/activate

cd src


python -m trainning.context_moe_train \
    --batch_size 32 \
    --epochs 300 \
    --num_experts 5 \
    --top_k 2 \
    --model_name mobilenetv3small_moe \
    --type_model MoE \
    --router_mode noisy \
    --use_context False \
    --temperature 1.0 \
    --dataset_name mixed_dataset
