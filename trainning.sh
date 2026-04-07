#!/bin/bash

# load conda
source venv/bin/activate

cd src

clear
PYTHONPATH=src python -m trainning.moe_train\
    --num_experts 2\
    --top_k 2\
    --num_epoch 300

clear
PYTHONPATH=src python -m trainning.moe_train\
    --num_experts 2\
    --top_k 2\
    --num_epoch 300

clear
PYTHONPATH=src python -m trainning.moe_train\
    --num_experts 2\
    --top_k 2\
    --num_epoch 300

clear
PYTHONPATH=src python -m trainning.moe_train\
    --num_experts 4\
    --top_k 1\
    --num_epoch 300

clear
PYTHONPATH=src python -m trainning.moe_train\
    --num_experts 4\
    --top_k 1\
    --num_epoch 300

clear
PYTHONPATH=src python -m trainning.moe_train\
    --num_experts 4\
    --top_k 1\
    --num_epoch 300

clear
PYTHONPATH=src python -m trainning.moe_train\
    --num_experts 4\
    --top_k 2\
    --num_epoch 300

clear
PYTHONPATH=src python -m trainning.moe_train\
    --num_experts 4\
    --top_k 2\
    --num_epoch 300

clear
PYTHONPATH=src python -m trainning.moe_train\
    --num_experts 4\
    --top_k 2\
    --num_epoch 300