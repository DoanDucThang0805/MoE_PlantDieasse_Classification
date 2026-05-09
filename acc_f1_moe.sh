#!/bin/bash
# Script to compute accuracy and F1 scores for all MoE checkpoints
source venv/bin/activate
source venv/Scripts/activate
cd src
clear
python -m benchmark.getaccvsf1 \
    --model_name mobilenetv3small_moe \
    --type_model moe_contextaware_temp0.5 \
    --dataset_name plantdoc \
    --export_to_csv \
    --csv_store_dir "/media/data/minhht/context_moe" \
    --csv_filename "moe_contextaware_temp0.5.csv"
    