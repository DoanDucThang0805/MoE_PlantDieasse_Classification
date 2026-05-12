#!/bin/bash

source venv/bin/activate

CHECKPOINT="/media/data/minhht/context_moe/checkpoints/plantdoc/moe_linearcontextaware_temp0.5/mobilenetv3small_moe/4_experts/top_2/seed_43/run_20260504-214159/best_checkpoint.pth"
OUTPUT="/media/data/minhht/context_moe/onnx/mobilenetv3small_moe.onnx"

python src/export_onnx/mobilenetv3small_moe.py \
    --output "$OUTPUT" \
    --num_classes 8 \
    --num_experts 4 \
    --top_k 2 \
    --context_dim 6 \
    --router_mode context_aware \
    --temperature 0.5 \
    --opset_version 18 \
    ${CHECKPOINT:+--checkpoint "$CHECKPOINT"}


