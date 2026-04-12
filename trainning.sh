#!/bin/bash

# load conda
source venv/bin/activate

cd src
clear

python -m trainning.moe_train \
    --type_model MoE_classweight \
    --num_experts 5 \
    --top_k 2 \
    --num_epochs 300