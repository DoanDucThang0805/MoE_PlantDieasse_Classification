#!/bin/bash
# Installation script for the application
source venv/bin/activate

cd src
clear
python -m inference.moe.context_aware_moe_inference \
    --model_name efficientnetv2m_moe \
    --type_model MoE \
    --run_time run_20260404-181413 \
    --run_time run_20260404-181134 \
    --dataset_name plantdoc \
    --use_context True \
    --router_mode context_aware
