"""
Evaluate Accuracy & macro-F1 drop for INT8 quantized models on PlantDoc dataset.

Compares FP32 vs INT8 ONNX models by running inference on the PlantDoc test set
and computing per-model accuracy and macro-F1, then reporting the drop.

Usage:
    python src/quantization/eval_int8_accuracy.py \
        --output_dir onnx/quantized \
        --dataset plantdoc \
        --csv_filename eval_int8_plantdoc.csv \
        --csv_store_dir results \
        --export_csv

Author: Context MoE Plant Disease Classification Team
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Default paths
ONNX_QUANTIZED_DIR = Path(__file__).resolve().parents[2] / "onnx" / "quantized"

MODEL_NAMES = [
    "efficientnetb0",
    "ghostnet",
    "mobilenetv3_small",
    "mobilevits",
    "mobilevitxs",
    "shufflenetv2",
    "squeezenet",
]


def build_test_loader(dataset_name: str, batch_size: int = 32) -> DataLoader:
    """
    Build a test DataLoader from the specified dataset.

    Args:
        dataset_name: "plantdoc" or "slif_tomato"
        batch_size: batch size for the test loader

    Returns:
        DataLoader for the test split
    """
    if dataset_name == "plantdoc":
        from dataset.plantdoc_dataset import build_datasets
    elif dataset_name == "slif_tomato":
        from dataset.slif_tomato_dataset import build_datasets
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'plantdoc' or 'slif_tomato'.")

    _, _, test_dataset = build_datasets(use_context=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    logger.info("Test set: %d samples (%s)", len(test_dataset), dataset_name)
    return test_loader


def evaluate_onnx(
    onnx_path: Path,
    test_loader: DataLoader,
) -> Tuple[float, float]:
    """
    Run inference on an ONNX model and return (accuracy, macro_f1).

    Works with both FP32 and INT8 quantized models.
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
        preds = np.argmax(outputs[0], axis=1).tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    mf1 = f1_score(all_labels, all_preds, average="macro")
    return acc, mf1


def get_model_size_mb(path: Path) -> float:
    """Model size in MB, including .data sidecar."""
    total = path.stat().st_size
    data_path = path.with_suffix(path.suffix + ".data")
    if data_path.exists():
        total += data_path.stat().st_size
    return total / (1024 ** 2)


class EvalInt8Accuracy:
    """
    Evaluate and compare FP32 vs INT8 ONNX model accuracy & macro-F1
    on a given test dataset.
    """

    def __init__(
        self,
        onnx_dir: Path,
        model_names: List[str],
        dataset_name: str = "plantdoc",
    ) -> None:
        self.onnx_dir = Path(onnx_dir)
        self.model_names = model_names
        self.dataset_name = dataset_name
        self.results: List[Dict] = []

    def run(self) -> pd.DataFrame:
        """Evaluate all models and return a comparison DataFrame."""
        test_loader = build_test_loader(self.dataset_name)
        self.results = []

        for name in self.model_names:
            fp32_path = self.onnx_dir / f"{name}_fp32.onnx"
            int8_path = self.onnx_dir / f"{name}_int8.onnx"

            if not fp32_path.exists() or not int8_path.exists():
                logger.warning(
                    "Skipping %s: FP32=%s INT8=%s",
                    name, fp32_path.exists(), int8_path.exists(),
                )
                continue

            logger.info("Evaluating %s (FP32) …", name)
            fp32_acc, fp32_f1 = evaluate_onnx(fp32_path, test_loader)

            logger.info("Evaluating %s (INT8) …", name)
            int8_acc, int8_f1 = evaluate_onnx(int8_path, test_loader)

            fp32_size = get_model_size_mb(fp32_path)
            int8_size = get_model_size_mb(int8_path)

            acc_drop = int8_acc - fp32_acc
            f1_drop = int8_f1 - fp32_f1
            size_reduction = ((fp32_size - int8_size) / fp32_size) * 100

            row = {
                "model": name,
                "fp32_accuracy": round(fp32_acc, 4),
                "fp32_macro_f1": round(fp32_f1, 4),
                "fp32_size_mb": round(fp32_size, 2),
                "int8_accuracy": round(int8_acc, 4),
                "int8_macro_f1": round(int8_f1, 4),
                "int8_size_mb": round(int8_size, 2),
                "acc_drop": round(acc_drop, 4),
                "f1_drop": round(f1_drop, 4),
                "size_reduction_pct": round(size_reduction, 1),
            }
            self.results.append(row)

            # Print per-model result
            print(f"\n{'='*65}")
            print(f"  {name} ({self.dataset_name})")
            print(f"{'='*65}")
            print(f"  {'':20s} | {'Accuracy':>10} | {'macro-F1':>10} | {'Size (MB)':>10}")
            print(f"  {'-'*57}")
            print(f"  {'FP32':20s} | {fp32_acc:>10.4f} | {fp32_f1:>10.4f} | {fp32_size:>10.2f}")
            print(f"  {'INT8':20s} | {int8_acc:>10.4f} | {int8_f1:>10.4f} | {int8_size:>10.2f}")
            print(f"  {'-'*57}")
            print(f"  {'Drop':20s} | {acc_drop:>+10.4f} | {f1_drop:>+10.4f} | {-size_reduction:>+9.1f}%")
            print(f"{'='*65}")

        df = pd.DataFrame(self.results)

        # Summary table
        if not df.empty:
            print(f"\n{'='*80}")
            print(f"  Summary: FP32 vs INT8 on {self.dataset_name}")
            print(f"{'='*80}")
            summary_cols = ["model", "fp32_macro_f1", "int8_macro_f1", "f1_drop", "size_reduction_pct"]
            print(df[summary_cols].to_string(index=False))
            print(f"{'='*80}")

        return df

    def export_to_csv(
        self,
        df: pd.DataFrame,
        filename: str,
        export_dir: Path,
    ) -> Path:
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        out = export_dir / filename
        df.to_csv(out, index=False)
        logger.info("✓ Results saved to: %s", out)
        return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate accuracy & macro-F1 drop for INT8 quantized ONNX models"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=ONNX_QUANTIZED_DIR,
        help="Directory containing *_fp32.onnx and *_int8.onnx files",
    )
    parser.add_argument(
        "--dataset", type=str, default="plantdoc",
        choices=["plantdoc", "slif_tomato"],
        help="Dataset to evaluate on",
    )
    parser.add_argument("--csv_store_dir", type=Path, default=Path("./results"))
    parser.add_argument("--csv_filename", type=str, default="eval_int8_plantdoc.csv")
    parser.add_argument("--export_csv", action="store_true")
    args = parser.parse_args()

    evaluator = EvalInt8Accuracy(
        onnx_dir=args.output_dir,
        model_names=MODEL_NAMES,
        dataset_name=args.dataset,
    )
    df = evaluator.run()

    if args.export_csv:
        evaluator.export_to_csv(df, args.csv_filename, args.csv_store_dir)


if __name__ == "__main__":
    main()
