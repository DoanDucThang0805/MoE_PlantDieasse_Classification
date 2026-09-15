"""
Shared utilities for ONNX static INT8 quantization.

Provides:
- CalibrationDataReader: feeds calibration samples to ONNX Runtime quantizer
- export_to_onnx_fp32(): PyTorch model → ONNX FP32
- quantize_static_int8(): ONNX FP32 → ONNX INT8 via static quantization
- evaluate_onnx_model(): run inference on ONNX model and compute macro-F1
- compare_and_report(): compare FP32 vs INT8 results and print summary table

Author: Context MoE Plant Disease Classification Team
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader as _CalibrationDataReaderBase,
    QuantFormat,
    QuantType,
    quantize_static,
)
from sklearn.metrics import f1_score, accuracy_score

# ---------------------------------------------------------------------------
# Ensure src/ is on sys.path so sibling packages (dataset, models, …) resolve
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


# =====================================================================
# 1. Calibration Data Reader
# =====================================================================

class CalibrationDataReader(_CalibrationDataReaderBase):
    """
    Feeds preprocessed image batches to the ONNX Runtime static quantizer.

    Each call to ``get_next()`` returns a dict ``{"input": np.ndarray}``
    with shape ``(1, 3, 224, 224)`` in float32.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        input_name: str = "input",
        max_samples: int = 200,
    ) -> None:
        self.input_name = input_name
        self.max_samples = max_samples

        # Pre-collect numpy arrays from the dataloader
        self._data: List[np.ndarray] = []
        count = 0
        for batch in dataloader:
            # dataset returns (image, label, context) — only need images
            images = batch[0]
            for i in range(images.size(0)):
                if count >= max_samples:
                    break
                self._data.append(images[i].unsqueeze(0).numpy())
                count += 1
            if count >= max_samples:
                break

        self._idx = 0
        logger.info("CalibrationDataReader: collected %d samples", len(self._data))

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self._idx >= len(self._data):
            return None
        feed = {self.input_name: self._data[self._idx]}
        self._idx += 1
        return feed

    def rewind(self) -> None:
        self._idx = 0


# =====================================================================
# 2. Load checkpoint into PyTorch model
# =====================================================================

def load_checkpoint(model: nn.Module, checkpoint_path: Path) -> nn.Module:
    """Load a ``best_checkpoint.pth`` file into *model* and return it."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    logger.info("✓ Loaded checkpoint: %s", checkpoint_path)
    return model


# =====================================================================
# 3. Export PyTorch → ONNX FP32
# =====================================================================

def export_to_onnx_fp32(
    model: nn.Module,
    output_path: Path,
    opset_version: int = 18,
) -> Path:
    """
    Trace *model* with a dummy input and save as ONNX FP32.

    Returns the resolved *output_path*.
    """
    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")          # export on CPU for compatibility
    model = model.to(device)
    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=opset_version,
        do_constant_folding=True,
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    logger.info("✓ Exported FP32 ONNX: %s", output_path)
    return output_path


# =====================================================================
# 4. Static INT8 Quantization
# =====================================================================

def quantize_static_int8(
    fp32_onnx_path: Path,
    int8_onnx_path: Path,
    calibration_reader: CalibrationDataReader,
) -> Path:
    """
    Apply ONNX Runtime static INT8 quantization.

    Uses ``QOperator`` format with ``QUInt8`` activations and ``QInt8`` weights.
    """
    fp32_onnx_path = Path(fp32_onnx_path)
    int8_onnx_path = Path(int8_onnx_path)
    int8_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    quantize_static(
        model_input=str(fp32_onnx_path),
        model_output=str(int8_onnx_path),
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    logger.info("✓ Quantized INT8 ONNX: %s", int8_onnx_path)
    return int8_onnx_path


# =====================================================================
# 5. Evaluate an ONNX model (FP32 or INT8) on a DataLoader
# =====================================================================

def evaluate_onnx_model(
    onnx_path: Path,
    test_loader: DataLoader,
) -> Tuple[float, float]:
    """
    Run inference with ONNX Runtime on *test_loader* and return
    ``(accuracy, macro_f1)`` as floats in [0, 1].
    """
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    all_preds: List[int] = []
    all_labels: List[int] = []

    for batch in test_loader:
        images, labels = batch[0], batch[1]
        outputs = session.run(None, {input_name: images.numpy()})
        logits = outputs[0]                          # (B, num_classes)
        preds = np.argmax(logits, axis=1).tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return accuracy, macro_f1


# =====================================================================
# 6. Model file size helper
# =====================================================================

def get_model_size_mb(path: Path) -> float:
    """Return total ONNX model size in MB (including .data sidecar)."""
    path = Path(path)
    total = path.stat().st_size
    data_path = path.with_suffix(path.suffix + ".data")
    if data_path.exists():
        total += data_path.stat().st_size
    return total / (1024 ** 2)


# =====================================================================
# 7. Compare & report
# =====================================================================

def compare_and_report(
    model_name: str,
    fp32_onnx_path: Path,
    int8_onnx_path: Path,
    test_loader: DataLoader,
) -> Dict[str, float]:
    """
    Evaluate both FP32 and INT8 ONNX models, print a comparison table,
    and return a results dict.
    """
    logger.info("Evaluating FP32 model …")
    fp32_acc, fp32_f1 = evaluate_onnx_model(fp32_onnx_path, test_loader)

    logger.info("Evaluating INT8 model …")
    int8_acc, int8_f1 = evaluate_onnx_model(int8_onnx_path, test_loader)

    fp32_size = get_model_size_mb(fp32_onnx_path)
    int8_size = get_model_size_mb(int8_onnx_path)

    f1_drop = int8_f1 - fp32_f1
    size_reduction = ((fp32_size - int8_size) / fp32_size) * 100.0

    # Pretty-print
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Static INT8 Quantization Results: {model_name}")
    print(sep)
    print(f"{'Model':<20} | {'Accuracy':>10} | {'macro-F1':>10} | {'Size (MB)':>10}")
    print("-" * 60)
    print(f"{'FP32 (original)':<20} | {fp32_acc:>10.4f} | {fp32_f1:>10.4f} | {fp32_size:>10.2f}")
    print(f"{'INT8 (quantized)':<20} | {int8_acc:>10.4f} | {int8_f1:>10.4f} | {int8_size:>10.2f}")
    print("-" * 60)
    print(f"{'mF1 Drop':<20} | {'':>10} | {f1_drop:>+10.4f} | {-size_reduction:>+9.1f}%")
    print(f"{sep}\n")

    return {
        "model_name": model_name,
        "fp32_accuracy": fp32_acc,
        "fp32_macro_f1": fp32_f1,
        "fp32_size_mb": fp32_size,
        "int8_accuracy": int8_acc,
        "int8_macro_f1": int8_f1,
        "int8_size_mb": int8_size,
        "f1_drop": f1_drop,
        "size_reduction_pct": size_reduction,
    }


# =====================================================================
# 8. Build calibration & test data loaders (from slif_tomato_dataset)
# =====================================================================

def build_calib_and_test_loaders(
    calib_size: int = 200,
    batch_size: int = 1,
    test_batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build a calibration DataLoader (subset of train set) and a test DataLoader
    from the SLIF Tomato Phase-I dataset.

    Args:
        calib_size: number of calibration samples from training set.
        batch_size: batch size for calibration loader (usually 1).
        test_batch_size: batch size for test loader.

    Returns:
        (calib_loader, test_loader)
    """
    from dataset.slif_tomato_dataset import build_datasets

    train_dataset, _, test_dataset = build_datasets(use_context=False)

    # Subset for calibration
    indices = list(range(min(calib_size, len(train_dataset))))
    calib_subset = Subset(train_dataset, indices)
    calib_loader = DataLoader(calib_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    logger.info(
        "DataLoaders ready: calib=%d samples, test=%d samples",
        len(calib_subset),
        len(test_dataset),
    )
    return calib_loader, test_loader
