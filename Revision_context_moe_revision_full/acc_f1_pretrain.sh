#!/bin/bash

# Navigate to project directory
cd /media/data/minhht/context_moe

# Activate virtual environment
source venv1/bin/activate

# Run the script
python src/benchmark/getaccvsf1pretrain.py \
    --checkpoint_dirs "/media/data/minhht/context_moe/checkpoints/slif_tomato_dataset_phase1/pretrain_models/squeezenet" \
    --csv_store_dir "./results" \
    --csv_filename "squeezenet_slif_tomato_dataset.csv" \
    --export_csv
