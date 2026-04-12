#!/bin/bash
# Installation script for the application
source venv/bin/activate

cd src
clear
python -m inference.moe.context_aware_moe_inference \
    --model_name mobilenetv3small_moe \
    --type_model noisy_moe \
    --num_experts 5 \
    --top_k 2 \
    --run_time run_20260411-120102 \
    --dataset_name mixed_dataset \
    --router_mode noisy \
    --no_context
