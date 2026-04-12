#!/bin/bash
# Inference script

# Activate virtual environment
source venv/bin/activate

# Move to project root (nếu script nằm ở root thì giữ nguyên)
PROJECT_ROOT=$(pwd)

# Default parameters
MODEL_NAME="mobilenetv3_small"
MODEL_TYPE="pretrain_weight"
RUN_TIME="run_20260411-145522"
DATASET_NAME="mixed_dataset"
BATCH_SIZE=32


clear
PYTHONPATH=$PROJECT_ROOT/src python -m inference.pretrained.inference \
    --model_name $MODEL_NAME \
    --model_type $MODEL_TYPE \
    --run_time $RUN_TIME \
    --dataset_name $DATASET_NAME \
    --batch_size $BATCH_SIZE
