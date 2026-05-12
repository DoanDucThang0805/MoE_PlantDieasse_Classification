#!/bin/bash

source venv/bin/activate

CHECKPOINT=""
OUTPUT="onnx/mobilenetv3small_moe.onnx"

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


