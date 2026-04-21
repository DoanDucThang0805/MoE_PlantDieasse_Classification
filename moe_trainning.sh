#!/bin/bash

# load conda
source venv/bin/activate

cd src

python -m trainning.context_moe_train \
    --batch_size 32 \
    --epochs 300 \
    --num_experts 4 \
    --top_k 2 \
    --model_name mobilenetv3small_moe \
    --type_model moe_temp0.5 \
    --router_mode context_aware \
    --temperature 1.0
