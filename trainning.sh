#!/bin/bash

# load conda
# source venv/bin/activate
source venv/Scripts/activate

cd src
clear

python -m trainning.moe_train \
    --seed 42 \
    --num_experts 4 \
    --top_k 2 \
    --num_epochs 300 \
    --batch_size 32 \
    --lr 0.001 \
    --weight_decay 0.001 \
    --moe_alpha 0.05 \
    --lambda_ortho 0.001 \
    --ortho_warmup_epochs 10 \
    --temperature 0.5 \
    --type_model moe_0.5temp_ortho0.001

python -m trainning.moe_train \
    --seed 42 \
    --num_experts 4 \
    --top_k 2 \
    --num_epochs 300 \
    --batch_size 32 \
    --lr 0.001 \
    --weight_decay 0.001 \
    --moe_alpha 0.05 \
    --lambda_ortho 0.001 \
    --ortho_warmup_epochs 10 \
    --temperature 0.5 \
    --type_model moe_0.5temp_ortho0.001

python -m trainning.moe_train \
    --seed 42 \
    --num_experts 4 \
    --top_k 2 \
    --num_epochs 300 \
    --batch_size 32 \
    --lr 0.001 \
    --weight_decay 0.001 \
    --moe_alpha 0.05 \
    --lambda_ortho 0.001 \
    --ortho_warmup_epochs 10 \
    --temperature 0.5 \
    --type_model moe_0.5temp_ortho0.001

python -m trainning.moe_train \
    --seed 42 \
    --num_experts 4 \
    --top_k 2 \
    --num_epochs 300 \
    --batch_size 32 \
    --lr 0.001 \
    --weight_decay 0.001 \
    --moe_alpha 0.05 \
    --lambda_ortho 0.005 \
    --ortho_warmup_epochs 10 \
    --temperature 0.5 \
    --type_model moe_0.5temp_ortho0.005

python -m trainning.moe_train \
    --seed 42 \
    --num_experts 4 \
    --top_k 2 \
    --num_epochs 300 \
    --batch_size 32 \
    --lr 0.001 \
    --weight_decay 0.001 \
    --moe_alpha 0.05 \
    --lambda_ortho 0.005 \
    --ortho_warmup_epochs 10 \
    --temperature 0.5 \
    --type_model moe_0.5temp_ortho0.005

python -m trainning.moe_train \
    --seed 42 \
    --num_experts 4 \
    --top_k 2 \
    --num_epochs 300 \
    --batch_size 32 \
    --lr 0.001 \
    --weight_decay 0.001 \
    --moe_alpha 0.05 \
    --lambda_ortho 0.005 \
    --ortho_warmup_epochs 10 \
    --temperature 0.5 \
    --type_model moe_0.5temp_ortho0.005

python -m trainning.moe_train \
    --seed 42 \
    --num_experts 4 \
    --top_k 2 \
    --num_epochs 300 \
    --batch_size 32 \
    --lr 0.001 \
    --weight_decay 0.001 \
    --moe_alpha 0.05 \
    --lambda_ortho 0.005 \
    --ortho_warmup_epochs 10 \
    --temperature 0.5 \
    --type_model moe_0.5temp_ortho0.005

python -m trainning.moe_train \
    --seed 42 \
    --num_experts 4 \
    --top_k 2 \
    --num_epochs 300 \
    --batch_size 32 \
    --lr 0.001 \
    --weight_decay 0.001 \
    --moe_alpha 0.05 \
    --lambda_ortho 0.005 \
    --ortho_warmup_epochs 10 \
    --temperature 0.5 \
    --type_model moe_0.5temp_ortho0.005

python -m trainning.moe_train \
    --seed 42 \
    --num_experts 4 \
    --top_k 2 \
    --num_epochs 300 \
    --batch_size 32 \
    --lr 0.001 \
    --weight_decay 0.001 \
    --moe_alpha 0.05 \
    --lambda_ortho 0.003 \
    --ortho_warmup_epochs 10 \
    --temperature 0.5 \
    --type_model moe_0.5temp_ortho0.003

python -m trainning.moe_train \
    --seed 42 \
    --num_experts 4 \
    --top_k 2 \
    --num_epochs 300 \
    --batch_size 32 \
    --lr 0.001 \
    --weight_decay 0.001 \
    --moe_alpha 0.05 \
    --lambda_ortho 0.003 \
    --ortho_warmup_epochs 10 \
    --temperature 0.5 \
    --type_model moe_0.5temp_ortho0.003

python -m trainning.moe_train \
    --seed 42 \
    --num_experts 4 \
    --top_k 2 \
    --num_epochs 300 \
    --batch_size 32 \
    --lr 0.001 \
    --weight_decay 0.001 \
    --moe_alpha 0.05 \
    --lambda_ortho 0.003 \
    --ortho_warmup_epochs 10 \
    --temperature 0.5 \
    --type_model moe_0.5temp_ortho0.003

python -m trainning.moe_train \
    --seed 42 \
    --num_experts 4 \
    --top_k 2 \
    --num_epochs 300 \
    --batch_size 32 \
    --lr 0.001 \
    --weight_decay 0.001 \
    --moe_alpha 0.05 \
    --lambda_ortho 0.003 \
    --ortho_warmup_epochs 10 \
    --temperature 0.5 \
    --type_model moe_0.5temp_ortho0.003

python -m trainning.moe_train \
    --seed 42 \
    --num_experts 4 \
    --top_k 2 \
    --num_epochs 300 \
    --batch_size 32 \
    --lr 0.001 \
    --weight_decay 0.001 \
    --moe_alpha 0.05 \
    --lambda_ortho 0.003 \
    --ortho_warmup_epochs 10 \
    --temperature 0.5 \
    --type_model moe_0.5temp_ortho0.003