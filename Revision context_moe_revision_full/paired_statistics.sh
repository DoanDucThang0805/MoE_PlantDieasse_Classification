#!/bin/bash
# ============================================================
# Seed-Wise Paired Statistical Test
# Outputs: diagnostics/statistical_tests/paired_statistics.csv
# ============================================================
set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_CSV="${1:-$REPO_ROOT/diagnostics/statistical_tests/paired_statistics.csv}"

# Activate virtual environment
if [ -f "$REPO_ROOT/venv1/bin/activate" ]; then
    source "$REPO_ROOT/venv1/bin/activate"
elif [ -f "$REPO_ROOT/venv/bin/activate" ]; then
    source "$REPO_ROOT/venv/bin/activate"
fi

mkdir -p "$(dirname "$OUTPUT_CSV")"

echo "=========================================="
echo "  Seed-Wise Paired Statistical Test"
echo "=========================================="
echo "  Output CSV : $OUTPUT_CSV"
echo "  Seeds      : 42 43 44 45 46"
echo "  Primary    : MoE vs MobileNetV3-Small"
echo "               (PlantDoc + SLIF-Tomato)"
echo "  Exploratory: MoE vs EfficientNetB0,"
echo "               GhostNet, ShuffleNetV2,"
echo "               SqueezeNet (PlantDoc)"
echo "=========================================="
echo ""

cd "$REPO_ROOT/src"

python -m statistical_tests.paired_statistics \
    --output_csv "$OUTPUT_CSV"

echo ""
echo "Done. Results saved to: $OUTPUT_CSV"
