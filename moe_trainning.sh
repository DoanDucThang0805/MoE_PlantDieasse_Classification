#!/bin/bash

# MoE Configuration Experiments Runner
# Skip: 4 experts, top_k=2 (already done)

source venv/bin/activate
source venv/Scripts/activate
cd src

# Config
SEEDS=(45)

run_training() {
    local num_experts=$1
    local top_k=$2
    local seed=$3
    
    echo "🚀 Running: $num_experts experts, top_k=$top_k, seed=$seed"
    python -m trainning.moe_train \
        --type_model moe_linearcontextaware_temp0.5 \
        --num_experts $num_experts \
        --top_k $top_k \
        --router_mode context_aware \
        --batch_size 32 \
        --num_epochs 300 \
        --temperature 0.5 \
        --moe_alpha 0.05 \
        --use_context \
        --seed $seed
}


# 4 experts
for seed in "${SEEDS[@]}"; do
    run_training 4 2 $seed
done


echo "✅ Done!"
