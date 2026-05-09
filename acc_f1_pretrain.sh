#!/bin/bash

# Navigate to project directory
cd /media/data/minhht/context_moe

# Activate virtual environment
source venv1/bin/activate

# Run the script
python src/benchmark/getaccvsf1pretrain.py \
    --checkpoint_dirs "/media/data/minhht/context_moe/checkpoints/plantdoc/pretrain_weight/custom_mobilenetv3_smallv1" \
    --csv_store_dir "./results" \
    --csv_filename "custom_mobilenetv3_smallv1_results.csv" \
    --export_csv
