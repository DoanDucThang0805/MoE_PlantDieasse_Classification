#!/bin/bash
# Batch inference script for all MoE models
# Automatically discovers and runs inference on all model variants in:
# checkpoints/plantdoc/moe_contextaware_temp1.0/mobilenetv3small_moe

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
fi

cd src
clear

echo "=========================================="
echo "Batch Inference - All MoE Models"
echo "=========================================="
echo ""

DATASET_NAME="plantdoc"
MODEL_NAME="mobilenetv3small_moe"
TYPE_MODEL="moe_contextaware_temp0.5ImageNet1K"
NUM_EXPERTS=4
TOP_K=2
SEED=42
RUN_TIME="run_20260517-130117"


python -m inference.moe.inference \
    --model_name "$MODEL_NAME" \
    --type_model "$TYPE_MODEL" \
    --dataset_name "$DATASET_NAME" \
    --num_experts "$NUM_EXPERTS" \
    --top_k "$TOP_K" \
    --seed "$SEED" \
    --run_time "$RUN_TIME" \
    --use_context \
    --router_mode context_aware
