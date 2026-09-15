#!/bin/bash
# =============================================================================
# Static INT8 Quantization for all pretrained models
#
# This script runs ONNX static INT8 quantization for each pretrained model,
# using calibration data from the SLIF Tomato Phase-I dataset.
# Each model exports FP32 + INT8 ONNX files and reports macro-F1 drop.
#
# Output: onnx/quantized/{model_name}_fp32.onnx, {model_name}_int8.onnx
#
# Usage: Fill in the checkpoint paths below, then run:
#   bash quantize_pretrain.sh
# =============================================================================

set -e

cd /media/data/minhht/context_moe
source venv1/bin/activate

OUTPUT_DIR="onnx/quantized"
CALIB_SIZE=200

# =============================================================================
# CHECKPOINT PATHS — Fill in your checkpoint paths here
# Leave empty ("") to skip that model or use pretrained ImageNet weights only
# =============================================================================

CKPT_EFFICIENTNETB0=""
CKPT_GHOSTNET=""
CKPT_MOBILENETV3_SMALL=""
CKPT_MOBILEVITS=""
CKPT_MOBILEVITXS=""
CKPT_SHUFFLENETV2=""
CKPT_SQUEEZENET=""

# =============================================================================

echo "============================================================"
echo "  Static INT8 Quantization Pipeline"
echo "  Output: ${OUTPUT_DIR}/"
echo "  Calibration samples: ${CALIB_SIZE}"
echo "============================================================"

# ---- 1. EfficientNet-B0 ----
echo ""
echo "[1/7] Quantizing EfficientNet-B0 ..."
python src/quantization/quantize_efficientnetb0.py \
    --output_dir "${OUTPUT_DIR}" \
    --calib_size ${CALIB_SIZE} \
    ${CKPT_EFFICIENTNETB0:+--checkpoint "${CKPT_EFFICIENTNETB0}"}

# ---- 2. GhostNet ----
echo ""
echo "[2/7] Quantizing GhostNet ..."
python src/quantization/quantize_ghostnet.py \
    --output_dir "${OUTPUT_DIR}" \
    --calib_size ${CALIB_SIZE} \
    ${CKPT_GHOSTNET:+--checkpoint "${CKPT_GHOSTNET}"}

# ---- 3. MobileNetV3-Small ----
echo ""
echo "[3/7] Quantizing MobileNetV3-Small ..."
python src/quantization/quantize_mobilenetv3_small.py \
    --output_dir "${OUTPUT_DIR}" \
    --calib_size ${CALIB_SIZE} \
    ${CKPT_MOBILENETV3_SMALL:+--checkpoint "${CKPT_MOBILENETV3_SMALL}"}

# ---- 4. MobileViT-S ----
echo ""
echo "[4/7] Quantizing MobileViT-S ..."
python src/quantization/quantize_mobilevits.py \
    --output_dir "${OUTPUT_DIR}" \
    --calib_size ${CALIB_SIZE} \
    ${CKPT_MOBILEVITS:+--checkpoint "${CKPT_MOBILEVITS}"}

# ---- 5. MobileViT-XS ----
echo ""
echo "[5/7] Quantizing MobileViT-XS ..."
python src/quantization/quantize_mobilevitxs.py \
    --output_dir "${OUTPUT_DIR}" \
    --calib_size ${CALIB_SIZE} \
    ${CKPT_MOBILEVITXS:+--checkpoint "${CKPT_MOBILEVITXS}"}

# ---- 6. ShuffleNet-V2 ----
echo ""
echo "[6/7] Quantizing ShuffleNet-V2 ..."
python src/quantization/quantize_shufflenetv2.py \
    --output_dir "${OUTPUT_DIR}" \
    --calib_size ${CALIB_SIZE} \
    ${CKPT_SHUFFLENETV2:+--checkpoint "${CKPT_SHUFFLENETV2}"}

# ---- 7. SqueezeNet ----
echo ""
echo "[7/7] Quantizing SqueezeNet ..."
python src/quantization/quantize_squeezenet.py \
    --output_dir "${OUTPUT_DIR}" \
    --calib_size ${CALIB_SIZE} \
    ${CKPT_SQUEEZENET:+--checkpoint "${CKPT_SQUEEZENET}"}

echo ""
echo "============================================================"
echo "  All quantization complete!"
echo "  Output files in: ${OUTPUT_DIR}/"
echo "============================================================"
ls -lh "${OUTPUT_DIR}/"
