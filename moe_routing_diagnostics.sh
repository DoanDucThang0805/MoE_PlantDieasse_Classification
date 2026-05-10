#!/bin/bash

set -e

usage() {
    echo "Usage:"
    echo "  bash moe_routing_diagnostics.sh --checkpoint PATH --output_dir DIR [--batch_size N]"
    echo ""
    echo "Example:"
    echo "  bash moe_routing_diagnostics.sh \\"
    echo "    --checkpoint checkpoints/plantdoc/moe_contextaware_temp0.5/mobilenetv3small_moe/4_experts/top_2/seed_42/run_YYYYMMDD-HHMMSS/best_checkpoint.pth \\"
    echo "    --output_dir reports/plantdoc/moe_contextaware_temp0.5/mobilenetv3small_moe/4_experts/top_2/seed_42/routing_diagnostics \\"
    echo "    --batch_size 32"
}

CHECKPOINT="/media/data/minhht/context_moe/checkpoints/plantdoc/moe_linearcontextaware_temp0.5/mobilenetv3small_moe/4_experts/top_2/seed_43/run_20260504-214159/best_checkpoint.pth"
OUTPUT_DIR="/media/data/minhht/context_moe/diagnostics"
BATCH_SIZE=32
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
        --batch_size)
            BATCH_SIZE="$2"
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

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

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

if [ -f "venv1/bin/activate" ]; then
    source venv1/bin/activate
elif [ -f "venv1/Scripts/activate" ]; then
    source venv1/Scripts/activate
fi

mkdir -p "$OUTPUT_DIR_ABS"

cd src

python -m diagnostics.moe_routing_diagnostics \
    --checkpoint "$CHECKPOINT_ABS" \
    --output_dir "$OUTPUT_DIR_ABS" \
    --batch_size "$BATCH_SIZE"

echo "Done. Results saved to: $OUTPUT_DIR_ABS"
