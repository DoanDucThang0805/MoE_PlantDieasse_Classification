#!/bin/bash

# load conda
# source venv/bin/activate
source venv/Scripts/activate

cd src
clear

python -m trainning.moe_train \
    --seed 43 \
    --num_experts 4 \
    --top_k 2 \
    --num_epochs 300 \
    --batch_size 32 \
    --lr 0.001 \
    --weight_decay 0.001 \
    --moe_alpha 0.05 \
    --lambda_ortho 0.01 \
    --temperature 0.5 \
    --type_model moe_0.5temperature

python -m trainning.moe_train \
    --seed 44 \
    --num_experts 4 \
    --top_k 2 \
    --num_epochs 300 \
    --batch_size 32 \
    --lr 0.001 \
    --weight_decay 0.001 \
    --moe_alpha 0.05 \
    --lambda_ortho 0.01 \
    --temperature 0.5 \
    --type_model moe_0.5temperature