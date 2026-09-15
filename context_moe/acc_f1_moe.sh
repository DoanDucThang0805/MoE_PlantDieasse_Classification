#!/bin/bash
# Script to compute accuracy and F1 scores for all MoE checkpoints
source venv1/bin/activate
source venv/Scripts/activate
cd src
clear
python -m benchmark.getaccvsf1moe \
    --model_name mobilenetv3small_moe \
    --type_model moe_linearcontextaware_temp0.5 \
    --dataset_name slif_tomato_dataset_phase1 \
    --export_to_csv \
    --csv_store_dir "/media/data/minhht/context_moe" \
    --csv_filename "moe_linearcontextaware_temp0.5_slif_tomato_dataset.csv"
    