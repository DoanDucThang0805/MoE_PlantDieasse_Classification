"""
Edge Benchmark for INT8 Quantized ONNX Models.

Measures model size, CPU inference time, and peak memory usage for
FP32 vs INT8 quantized ONNX models side-by-side.

Reuses the same methodology as src/benchmark/edge_benchmark.py but
targets the quantized models in onnx/quantized/.

Usage:
    python src/quantization/edge_benchmark_int8.py \
        --output_dir onnx/quantized \
        --csv_filename edge_benchmark_int8_results.csv \
        --csv_store_dir results

Author: Context MoE Plant Disease Classification Team
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import onnxruntime as ort
import pandas as pd
import psutil
from memory_profiler import memory_usage

# Default INT8 quantized ONNX model directory
ONNX_QUANTIZED_DIR = Path(__file__).resolve().parents[2] / "onnx" / "quantized"

# Models to benchmark — each entry maps to {name}_fp32.onnx and {name}_int8.onnx
MODEL_NAMES = [
    "efficientnetb0",
    "ghostnet",
    "mobilenetv3_small",
    "mobilevits",
    "mobilevitxs",
    "shufflenetv2",
    "squeezenet",
]


class EdgeBenchmarkInt8:
    """
    Benchmark FP32 vs INT8 quantized ONNX models on CPU.

    For each model, measures:
    - Model size (MB)
    - Average inference time (ms)
    - Peak memory usage (MB)
    """

    def __init__(self, onnx_dir: Path, model_names: List[str]) -> None:
        self.onnx_dir = Path(onnx_dir)
        self.model_names = model_names
        self.results: List[Dict] = []

    def _load_session(self, onnx_path: Path) -> ort.InferenceSession:
        """Load an ONNX model into an ORT InferenceSession."""
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )

    def _create_feed_dict(
        self,
        session: ort.InferenceSession,
        batch_size: int = 1,
    ) -> Dict[str, np.ndarray]:
        """Build a feed dict from session input metadata."""
        feed = {}
        for inp in session.get_inputs():
            # Resolve shape: replace dynamic dims with concrete values
            shape = []
            for dim in inp.shape:
                if isinstance(dim, int):
                    shape.append(dim)
                else:
                    shape.append(batch_size)
            feed[inp.name] = np.random.randn(*shape).astype(np.float32)
        return feed

    def _model_size_mb(self, path: Path) -> float:
        """Return ONNX file size in MB (including .data sidecar)."""
        total = path.stat().st_size
        data_path = path.with_suffix(path.suffix + ".data")
        if data_path.exists():
            total += data_path.stat().st_size
        return total / (1024 ** 2)

    def _peak_memory_mb(
        self,
        session: ort.InferenceSession,
        feed_dict: Dict[str, np.ndarray],
    ) -> float:
        """Measure peak RSS increase during a single inference call."""
        gc.collect()
        process = psutil.Process(os.getpid())
        baseline = process.memory_info().rss / (1024 ** 2)
        peak = memory_usage(
            (session.run, [None, feed_dict]),
            interval=0.01,
            max_usage=True,
            include_children=True,
        )
        return round(peak - baseline, 4)

    def _inference_time_ms(
        self,
        session: ort.InferenceSession,
        feed_dict: Dict[str, np.ndarray],
        num_runs: int = 100,
        num_warmup: int = 10,
    ) -> float:
        """Average inference time in milliseconds."""
        for _ in range(num_warmup):
            session.run(None, feed_dict)

        t0 = time.perf_counter()
        for _ in range(num_runs):
            session.run(None, feed_dict)
        t1 = time.perf_counter()
        return round(((t1 - t0) / num_runs) * 1000, 5)

    def _bench_one(self, onnx_path: Path, label: str) -> Optional[Dict]:
        """Run all benchmarks for a single ONNX file."""
        if not onnx_path.exists():
            print(f"  ⚠ Skipping {label}: {onnx_path} not found")
            return None

        session = self._load_session(onnx_path)
        feed = self._create_feed_dict(session)

        size_mb = self._model_size_mb(onnx_path)
        peak_mem = self._peak_memory_mb(session, feed)
        latency = self._inference_time_ms(session, feed)

        return {
            "model_name": label,
            "model_size_mb": round(size_mb, 4),
            "cpu_peak_memory_mb": peak_mem,
            "cpu_inference_time_ms": latency,
        }

    def run_benchmarks(self) -> pd.DataFrame:
        """Run benchmarks for all FP32 + INT8 model pairs."""
        self.results = []

        for name in self.model_names:
            fp32_path = self.onnx_dir / f"{name}_fp32.onnx"
            int8_path = self.onnx_dir / f"{name}_int8.onnx"

            print(f"\n{'='*60}")
            print(f"  Benchmarking: {name}")
            print(f"{'='*60}")

            # FP32
            fp32_row = self._bench_one(fp32_path, f"{name}_fp32")
            if fp32_row:
                self.results.append(fp32_row)
                print(
                    f"  FP32: size={fp32_row['model_size_mb']} MB, "
                    f"mem={fp32_row['cpu_peak_memory_mb']} MB, "
                    f"latency={fp32_row['cpu_inference_time_ms']} ms"
                )

            # INT8
            int8_row = self._bench_one(int8_path, f"{name}_int8")
            if int8_row:
                self.results.append(int8_row)
                print(
                    f"  INT8: size={int8_row['model_size_mb']} MB, "
                    f"mem={int8_row['cpu_peak_memory_mb']} MB, "
                    f"latency={int8_row['cpu_inference_time_ms']} ms"
                )

            # Comparison
            if fp32_row and int8_row:
                size_red = (1 - int8_row["model_size_mb"] / fp32_row["model_size_mb"]) * 100
                speedup = fp32_row["cpu_inference_time_ms"] / max(int8_row["cpu_inference_time_ms"], 1e-6)
                print(f"  → Size reduction: {size_red:.1f}%,  Speedup: {speedup:.2f}x")

        return self._to_dataframe()

    def _to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.results,
            columns=[
                "model_name",
                "model_size_mb",
                "cpu_peak_memory_mb",
                "cpu_inference_time_ms",
            ],
        )

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
        print(f"\n✓ Results saved to: {out}")
        return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Edge benchmark for FP32 vs INT8 quantized ONNX models"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=ONNX_QUANTIZED_DIR,
        help="Directory containing *_fp32.onnx and *_int8.onnx files",
    )
    parser.add_argument("--csv_store_dir", type=Path, default=Path("./results"))
    parser.add_argument("--csv_filename", type=str, default="edge_benchmark_int8_results.csv")
    parser.add_argument("--export_csv", action="store_true")
    args = parser.parse_args()

    bench = EdgeBenchmarkInt8(onnx_dir=args.output_dir, model_names=MODEL_NAMES)
    df = bench.run_benchmarks()
    print("\n" + df.to_string(index=False))

    if args.export_csv:
        bench.export_to_csv(df, args.csv_filename, args.csv_store_dir)


if __name__ == "__main__":
    main()
