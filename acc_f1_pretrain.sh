#!/bin/bash

# Navigate to project directory
cd /media/data/minhht/context_moe

# Activate virtual environment
source venv1/bin/activate

# Run the script
python src/benchmark/getaccvsf1pretrain.py \
    --checkpoint_dirs "/media/icnlab/Data/Thang/plan_dieases/context_moe/checkpoints/plantdoc/pretrain_weight/efficientnetb0" \
    --csv_store_dir "./results" \
    --csv_filename "efficientnetb0_results.csv" \
    --export_csv
