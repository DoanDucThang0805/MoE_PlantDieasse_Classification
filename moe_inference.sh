#!/bin/bash
# Installation script for the application
source venv/bin/activate

cd src
clear
python -m inference.moe.context_aware_moe_inference \
    --model_name efficientnetv2m_moe \
    --type_model MoE \
<<<<<<< HEAD
    --run_time run_20260404-181413 \
=======
    --run_time run_20260404-181134 \
>>>>>>> 4a9830f743ef988c2b15587ec53b7404da71841e
    --dataset_name plantdoc \
    --use_context True \
    --router_mode context_aware
