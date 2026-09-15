"""
Static INT8 Quantization for GhostNet.

Pipeline:
    1. Load GhostNet (timm) + checkpoint weights
    2. Export → ONNX FP32
    3. Calibrate + quantize → ONNX INT8
    4. Evaluate macro-F1 drop on test set

Usage:
    python src/quantization/quantize_ghostnet.py \
        --checkpoint <path_to_best_checkpoint.pth> \
        --output_dir onnx/quantized \
        --calib_size 200

Author: Context MoE Plant Disease Classification Team
"""

import argparse
import copy
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import torch.nn as nn

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantization.quantize_utils import (
    CalibrationDataReader,
    build_calib_and_test_loaders,
    compare_and_report,
    export_to_onnx_fp32,
    load_checkpoint,
    quantize_static_int8,
)

MODEL_NAME = "ghostnet"


def build_model(num_classes: int = 8) -> nn.Module:
    """Create a fresh GhostNet instance via timm."""
    import timm

    model = timm.create_model(
        "ghostnet_100",
        pretrained=True,
        num_classes=num_classes,
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Static INT8 quantization: {MODEL_NAME}")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to best_checkpoint.pth")
    parser.add_argument("--output_dir", type=Path, default=Path("onnx/quantized"),
                        help="Directory to save ONNX files")
    parser.add_argument("--calib_size", type=int, default=200,
                        help="Number of calibration samples")
    parser.add_argument("--num_classes", type=int, default=8)
    args = parser.parse_args()

    # ---- 1. Build model & load weights ----
    model = build_model(num_classes=args.num_classes)
    if args.checkpoint is not None:
        model = load_checkpoint(model, args.checkpoint)

    # ---- 2. Export to ONNX FP32 ----
    fp32_path = args.output_dir / f"{MODEL_NAME}_fp32.onnx"
    export_to_onnx_fp32(model, fp32_path)

    # ---- 3. Calibration + static INT8 quantization ----
    calib_loader, test_loader = build_calib_and_test_loaders(
        calib_size=args.calib_size,
    )
    calibration_reader = CalibrationDataReader(
        dataloader=calib_loader,
        input_name="input",
        max_samples=args.calib_size,
    )

    int8_path = args.output_dir / f"{MODEL_NAME}_int8.onnx"
    quantize_static_int8(fp32_path, int8_path, calibration_reader)

    # ---- 4. Evaluate & report ----
    results = compare_and_report(MODEL_NAME, fp32_path, int8_path, test_loader)


if __name__ == "__main__":
    main()
