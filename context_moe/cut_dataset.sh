#!/bin/bash

# Dataset Cutter - Extract a subset of tomato disease dataset

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <percentage> <output_dir> [seed]"
    echo ""
    echo "Examples:"
    echo "  $0 20 tomato_20pct          # Extract 20% to tomato_20pct/"
    echo "  $0 50 tomato_50pct 42       # Extract 50% to tomato_50pct/ with seed 42"
    echo ""
    exit 1
fi

PERCENTAGE=$1
OUTPUT_DIR=$2
SEED=${3:-42}
SOURCE_DIR="/media/data/minhht/context_moe/data/tomato_only"

# Validate percentage
if ! [[ "$PERCENTAGE" =~ ^[0-9]+(\.[0-9]+)?$ ]] || (( $(echo "$PERCENTAGE <= 0 || $PERCENTAGE > 100" | bc -l) )); then
    echo "❌ Error: Percentage must be between 0 and 100"
    exit 1
fi

# Get absolute path for output
OUTPUT_PATH="/media/data/minhht/context_moe/data/$OUTPUT_DIR"

echo ""
echo "============================================================================"
echo "Dataset Cutter - Tomato Disease Dataset"
echo "============================================================================"
echo "Source:     $SOURCE_DIR"
echo "Output:     $OUTPUT_PATH"
echo "Percentage: $PERCENTAGE%"
echo "Seed:       $SEED"
echo "============================================================================"
echo ""

# Run Python script
cd /media/data/minhht/context_moe/src
python -m utils.cut_dataset \
    --source_dir "$SOURCE_DIR" \
    --output_dir "$OUTPUT_PATH" \
    --percentage "$PERCENTAGE" \
    --seed "$SEED"
