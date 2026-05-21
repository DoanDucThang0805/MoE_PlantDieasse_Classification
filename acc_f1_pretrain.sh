#!/bin/bash

# Navigate to project directory
cd /media/data/minhht/context_moe

# Activate virtual environment
source venv1/bin/activate

# Run the script
python src/benchmark/getaccvsf1pretrain.py \
    --checkpoint_dirs "/media/data/minhht/context_moe/checkpoints/plantdoc/pretrain_models/mobilenetv3_small" \
    --csv_store_dir "./results" \
    --csv_filename "mobilenetv3_small.csv" \
    --export_csv
