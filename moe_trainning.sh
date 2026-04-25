#!/bin/bash

# MoE Configuration Experiments Runner
# Skip: 4 experts, top_k=2 (already done)

source venv/bin/activate
source venv/Scripts/activate
cd src

# Config
SEEDS=(42 43 44 45 46)

run_training() {
    local num_experts=$1
    local top_k=$2
    local seed=$3
    
    echo "🚀 Running: $num_experts experts, top_k=$top_k, seed=$seed"
    python -m trainning.moe_train \
        --type_model moe_contextaware_temp0.7 \
        --num_experts $num_experts \
        --top_k $top_k \
        --router_mode context_aware \
        --batch_size 32 \
        --num_epochs 300 \
        --temperature 0.7 \
        --moe_alpha 0.05 \
        --use_context \
        --seed $seed
}

# Experiments: 2,3,4,5 experts (top_k=1,2) + 6,7 (top_k=1,2,3) + 8 (top_k=1,2,3,4)

# 2 experts
for seed in "${SEEDS[@]}"; do
    run_training 2 1 $seed
    run_training 2 2 $seed
done

# 3 experts
for seed in "${SEEDS[@]}"; do
    run_training 3 1 $seed
    run_training 3 2 $seed
done

# 4 experts
for seed in "${SEEDS[@]}"; do
    run_training 4 1 $seed
    run_training 4 2 $seed
done

# 5 experts
for seed in "${SEEDS[@]}"; do
    run_training 5 1 $seed
    run_training 5 2 $seed
done

# 6 experts
for seed in "${SEEDS[@]}"; do
    run_training 6 1 $seed
    run_training 6 2 $seed
    run_training 6 3 $seed
done

# 7 experts
for seed in "${SEEDS[@]}"; do
    run_training 7 1 $seed
    run_training 7 2 $seed
    run_training 7 3 $seed
done

# 8 experts
for seed in "${SEEDS[@]}"; do
    run_training 8 1 $seed
    run_training 8 2 $seed
    run_training 8 3 $seed
    run_training 8 4 $seed
done

echo "✅ Done!"
