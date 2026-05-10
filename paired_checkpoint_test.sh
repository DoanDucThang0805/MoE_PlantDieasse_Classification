#!/bin/bash

set -e

usage() {
    echo "Usage:"
    echo "  bash paired_checkpoint_test.sh --model_a_name NAME --model_a_dir DIR --model_a_type TYPE \\"
    echo "    --model_b_name NAME --model_b_dir DIR --model_b_type TYPE --output_csv PATH [options]"
    echo ""
    echo "Model types:"
    echo "  moe, dense_multibranch, mobilenetv3_small, mobilenetv3_large,"
    echo "  widened_mlp_head, shufflenet, efficientnet_b4, resnet50"
    echo ""
    echo "Options:"
    echo "  --split SPLIT           Dataset split: train, validation, test (default: test)"
    echo "  --batch_size N          Batch size for inference (default: 32)"
    echo "  --seeds IDS             Comma-separated seeds (default: 42,43,44,45,46)"
    echo "  -h, --help              Show this help"
    echo ""
    echo "Example:"
    echo "  bash paired_checkpoint_test.sh \\"
    echo "    --model_a_name MoE_MLP \\"
    echo "    --model_a_dir checkpoints/plantdoc/moe_contextaware_temp0.5/mobilenetv3small_moe/4_experts/top_2 \\"
    echo "    --model_a_type moe \\"
    echo "    --model_b_name MoE_Linear \\"
    echo "    --model_b_dir checkpoints/plantdoc/moe_linearcontextaware_temp0.5/mobilenetv3small_moe/4_experts/top_2 \\"
    echo "    --model_b_type moe \\"
    echo "    --output_csv diagnostics/statistical_tests/moe_mlp_vs_linear.csv \\"
    echo "    --split test \\"
    echo "    --batch_size 32 \\"
    echo "    --seeds 42,43,44,45,46"
}

MODEL_A_NAME=""
MODEL_A_DIR=""
MODEL_A_TYPE=""
MODEL_B_NAME=""
MODEL_B_DIR=""
MODEL_B_TYPE=""
OUTPUT_CSV=""
SPLIT="test"
BATCH_SIZE=32
SEEDS="42,43,44,45,46"
REPO_ROOT="$(pwd)"

while [ $# -gt 0 ]; do
    case "$1" in
        --model_a_name)
            MODEL_A_NAME="$2"
            shift 2
            ;;
        --model_a_dir)
            MODEL_A_DIR="$2"
            shift 2
            ;;
        --model_a_type)
            MODEL_A_TYPE="$2"
            shift 2
            ;;
        --model_b_name)
            MODEL_B_NAME="$2"
            shift 2
            ;;
        --model_b_dir)
            MODEL_B_DIR="$2"
            shift 2
            ;;
        --model_b_type)
            MODEL_B_TYPE="$2"
            shift 2
            ;;
        --output_csv)
            OUTPUT_CSV="$2"
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
        --seeds)
            SEEDS="$2"
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

for required in MODEL_A_NAME MODEL_A_DIR MODEL_A_TYPE MODEL_B_NAME MODEL_B_DIR MODEL_B_TYPE OUTPUT_CSV; do
    if [ -z "${!required}" ]; then
        echo "ERROR: --$(echo "$required" | tr '[:upper:]' '[:lower:]') is required"
        usage
        exit 1
    fi
done

case "$SPLIT" in
    train|validation|test)
        ;;
    *)
        echo "ERROR: --split must be one of: train, validation, test"
        exit 1
        ;;
esac

resolve_path() {
    local input_path="$1"
    case "$input_path" in
        /*)
            echo "$input_path"
            ;;
        *)
            echo "$REPO_ROOT/$input_path"
            ;;
    esac
}

MODEL_A_DIR_ABS="$(resolve_path "$MODEL_A_DIR")"
MODEL_B_DIR_ABS="$(resolve_path "$MODEL_B_DIR")"
OUTPUT_CSV_ABS="$(resolve_path "$OUTPUT_CSV")"

if [ ! -d "$MODEL_A_DIR_ABS" ]; then
    echo "ERROR: model_a_dir not found: $MODEL_A_DIR_ABS"
    exit 1
fi

if [ ! -d "$MODEL_B_DIR_ABS" ]; then
    echo "ERROR: model_b_dir not found: $MODEL_B_DIR_ABS"
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

mkdir -p "$(dirname "$OUTPUT_CSV_ABS")"

echo "=========================================="
echo "Paired Checkpoint Statistical Test"
echo "=========================================="
echo "Model A    : $MODEL_A_NAME ($MODEL_A_TYPE)"
echo "Model A dir: $MODEL_A_DIR_ABS"
echo "Model B    : $MODEL_B_NAME ($MODEL_B_TYPE)"
echo "Model B dir: $MODEL_B_DIR_ABS"
echo "Output CSV : $OUTPUT_CSV_ABS"
echo "Split      : $SPLIT"
echo "Batch size : $BATCH_SIZE"
echo "Seeds      : $SEEDS"
echo "=========================================="

cd src

python -m statistical_tests.paired_checkpoint_test \
    --model_a_name "$MODEL_A_NAME" \
    --model_a_dir "$MODEL_A_DIR_ABS" \
    --model_a_type "$MODEL_A_TYPE" \
    --model_b_name "$MODEL_B_NAME" \
    --model_b_dir "$MODEL_B_DIR_ABS" \
    --model_b_type "$MODEL_B_TYPE" \
    --output_csv "$OUTPUT_CSV_ABS" \
    --split "$SPLIT" \
    --batch_size "$BATCH_SIZE" \
    --seeds "$SEEDS"

echo "Done. Results saved to: $OUTPUT_CSV_ABS"
