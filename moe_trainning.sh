#!/bin/bash

# load conda
source venv/bin/activate

cd src

python -m trainning.moe_train \
    --type_model moe_contextaware_temp0.5 \
    --num_experts 4 \
    --top_k 2 \
    --router_mode context_aware \
    --batch_size 32 \
    --num_epochs 300 \
    --temperature 0.5 \
    --moe_alpha 0.05 \
    --use_context \
    --seed 44 \