#!/bin/bash

set -e

usage() {
    echo "Usage:"
    echo "  bash moe_routing_diagnostics.sh --checkpoint PATH --output_dir DIR [options]"
    echo ""
    echo "Required:"
    echo "  --checkpoint PATH       Path to best_checkpoint.pth"
    echo "  --output_dir DIR        Directory to save outputs"
    echo ""
    echo "Options:"
    echo "  --split SPLIT           Dataset split: train, validation, test (default: test)"
    echo "  --batch_size N          Batch size for inference (default: 32)"
    echo "  --csv_name NAME         Output CSV file name (default: global_expert_usage.csv)"
    echo "  --plot_name NAME        Output plot file name (default: global_expert_usage.png)"
    echo "  -h, --help              Show this help"
    echo ""
    echo "Example:"
    echo "  bash moe_routing_diagnostics.sh \\"
    echo "    --checkpoint checkpoints/plantdoc/moe_contextaware_temp0.5/mobilenetv3small_moe/4_experts/top_2/seed_42/run_YYYYMMDD-HHMMSS/best_checkpoint.pth \\"
    echo "    --output_dir reports/plantdoc/moe_contextaware_temp0.5/mobilenetv3small_moe/4_experts/top_2/seed_42/routing_diagnostics \\"
    echo "    --split test \\"
    echo "    --batch_size 32 \\"
    echo "    --csv_name linear_global_expert_usage_seed42.csv \\"
    echo "    --plot_name linear_global_expert_usage_seed42.png"
}

CHECKPOINT="/media/data/minhht/context_moe/checkpoints/plantdoc/moe_linearcontextaware_temp0.5/mobilenetv3small_moe/4_experts/top_2/seed_43/run_20260504-214159/best_checkpoint.pth"
OUTPUT_DIR="/media/data/minhht/context_moe/diagnostics"
SPLIT="test"
BATCH_SIZE=32
CSV_NAME="linear_global_expert_usage.csv"
PLOT_NAME="linear_global_expert_usage.png"
REPO_ROOT="$(pwd)"

while [ $# -gt 0 ]; do
    case "$1" in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --csv_name)
            CSV_NAME="$2"
            shift 2
            ;;
        --plot_name)
            PLOT_NAME="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: --checkpoint is required"
    usage
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo "ERROR: --output_dir is required"
    usage
    exit 1
fi

case "$SPLIT" in
    train|validation|test)
        ;;
    *)
        echo "ERROR: --split must be one of: train, validation, test"
        exit 1
        ;;
esac

case "$CHECKPOINT" in
    /*)
        CHECKPOINT_ABS="$CHECKPOINT"
        ;;
    *)
        CHECKPOINT_ABS="$REPO_ROOT/$CHECKPOINT"
        ;;
esac

case "$OUTPUT_DIR" in
    /*)
        OUTPUT_DIR_ABS="$OUTPUT_DIR"
        ;;
    *)
        OUTPUT_DIR_ABS="$REPO_ROOT/$OUTPUT_DIR"
        ;;
esac

if [ ! -f "$CHECKPOINT_ABS" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT_ABS"
    exit 1
fi

if [ -f "venv1/bin/activate" ]; then
    source venv1/bin/activate
elif [ -f "venv1/Scripts/activate" ]; then
    source venv1/Scripts/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
fi

mkdir -p "$OUTPUT_DIR_ABS"

echo "=========================================="
echo "MoE Global Expert Usage Diagnostics"
echo "=========================================="
echo "Checkpoint : $CHECKPOINT_ABS"
echo "Output dir : $OUTPUT_DIR_ABS"
echo "Split      : $SPLIT"
echo "Batch size : $BATCH_SIZE"
echo "CSV        : $CSV_NAME"
echo "Plot       : $PLOT_NAME"
echo "=========================================="

cd src

python -m diagnostics.moe_routing_diagnostics \
    --checkpoint "$CHECKPOINT_ABS" \
    --output_dir "$OUTPUT_DIR_ABS" \
    --split "$SPLIT" \
    --batch_size "$BATCH_SIZE" \
    --csv_name "$CSV_NAME" \
    --plot_name "$PLOT_NAME"

echo "Done. Results saved to: $OUTPUT_DIR_ABS"
