#!/bin/bash

# load conda
source venv/bin/activate

cd src

python -m trainning.moe_train \
    --type-model MoE_classweight \
    --num-experts 5 \
    --top-k 2 \
    --num-epochs 300