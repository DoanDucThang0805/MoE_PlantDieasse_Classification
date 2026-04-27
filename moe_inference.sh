#!/bin/bash
# Batch inference script for all MoE checkpoints
source venv/bin/activate
source venv/Scripts/activate
cd src
clear
python -m inference.moe.context_aware_moe_inference \
    --model_name mobilenetv3small_moe \
    --type_model moe_contextaware_temp0.5 \
    --run_time run_20260423-152952 \
    --dataset_name plantdoc \
    --use_context \
    --router_mode context_aware \
    --num_experts 4 \
    --top_k 2 \
    --seed 45 \