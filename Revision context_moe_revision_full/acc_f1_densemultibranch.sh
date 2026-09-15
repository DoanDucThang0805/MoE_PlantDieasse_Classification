#!/bin/bash

source venv1/bin/activate
cd src

python -m benchmark.getaccvsf1densemultibranch \
    --checkpoint_dirs ../checkpoints/plantdoc/dense_multibranch/mobilenetv3small_dense_multibranch \
    --csv_store_dir ../results \
    --csv_filename mobilenetv3small_dense_multibranch_results.csv \
    --export_csv
            
