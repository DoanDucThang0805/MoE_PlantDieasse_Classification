#!/bin/bash
# Installation script for the application
source venv/bin/activate
source venv/Scripts/activate
cd src
clear
python -m inference.moe.context_aware_moe_inference \
    --model_name mobilenetv3small_moe \
    --type_model moe_contextaware_temp0.7 \
    --run_time run_20260425-131207 \
    --dataset_name plantdoc \
    --use_context \
    --router_mode context_aware \
    --num_experts 2 \
    --top_k 1 \
    --seed 42 \