#!/bin/bash

# load conda
source venv/bin/activate

cd src

# Sweep num_experts from 2 to 8
for num_experts in {2..8}; do
    # For each num_experts, sweep top_k from 1 to num_experts
    for top_k in $(seq 1 $num_experts); do
        # For each top_k, run 3 times
        for run in {1..3}; do
            clear
            python -m trainning.context_moe_train \
                --batch_size 32 \
                --epochs 300 \
                --num_experts $num_experts \
                --top_k $top_k \
                --model_name mobilenetv3small_moe \
                --type_model MoE \
                --router_mode context_aware \
                --use_context True
        done
    done
done
