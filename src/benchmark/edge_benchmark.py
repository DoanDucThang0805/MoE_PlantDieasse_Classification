import gc
import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import onnxruntime as ort
import pandas as pd
import psutil
from memory_profiler import memory_usage

onnx_root_dir = Path(__file__).parent.parent.parent / "onnx"
ONNX_MODEL_PATHS = {
    "mobilenetv3_small_baseline": onnx_root_dir / "mobilenetv3small.onnx",
    "mobilenetv3_small_widense": onnx_root_dir / "mobilenetv3small_widense.onnx",
    "mobilenetv3_small_dense_multibranch": onnx_root_dir / "mobilenetv3small_dense_multibranch.onnx",
    "mobilenetv3_small_moe": onnx_root_dir / "mobilenetv3small_moe.onnx",
}


class EdgeBenchmark:
    def __init__(self, onnx_model_paths: Dict[str, Path]):
        self.onnx_model_paths = onnx_model_paths
        self.results = []

    def load_onnx_session(self, model_name: str) -> ort.InferenceSession:
        if model_name not in self.onnx_model_paths:
            raise ValueError(f"Model {model_name} not found in ONNX registry.")

        model_path = Path(self.onnx_model_paths[model_name])
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        return ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

    def create_dummy_input(self, batch_size: int = 1) -> np.ndarray:
        return np.random.randn(batch_size, 3, 224, 224).astype(np.float32)

    def create_dummy_context(
        self,
        batch_size: int = 1,
        context_dim: int = 6,
    ) -> np.ndarray:
        return np.random.randn(batch_size, context_dim).astype(np.float32)

    def create_feed_dict(
        self,
        session: ort.InferenceSession,
        dummy_input: np.ndarray,
        dummy_context: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        feed_dict = {}
        input_names = [input_meta.name for input_meta in session.get_inputs()]

        if "image" in input_names:
            feed_dict["image"] = dummy_input
        else:
            feed_dict[input_names[0]] = dummy_input

        if len(input_names) > 1:
            if dummy_context is None:
                dummy_context = self.create_dummy_context(
                    batch_size=dummy_input.shape[0]
                )
            context_name = "context" if "context" in input_names else input_names[1]
            feed_dict[context_name] = dummy_context

        return feed_dict

    def caculate_model_size(self, model_path: Path) -> float:
        """
        Return ONNX model size in MB.

        If the model uses external data, include files such as *.onnx.data.
        """
        model_path = Path(model_path)
        total_size_bytes = model_path.stat().st_size

        external_data_path = model_path.with_suffix(model_path.suffix + ".data")
        if external_data_path.exists():
            total_size_bytes += external_data_path.stat().st_size

        total_size_mb = total_size_bytes / (1024 ** 2)
        return total_size_mb

    def calculate_memory_peak_in_cpu(self, session, feed_dict) -> float:
        gc.collect()
        process = psutil.Process(os.getpid())
        baseline = process.memory_info().rss / (1024 ** 2)  # đo TRƯỚC khi gọi memory_usage

        peak = memory_usage(
            (session.run, [None, feed_dict]),
            interval=0.01,
            max_usage=True,
            include_children=True,
        )
        return round(peak - baseline, 4)

    def caculate_inference_time_cpu(
        self,
        session: ort.InferenceSession,
        feed_dict: Dict[str, np.ndarray],
        num_runs: int = 100,
        num_warmup: int = 10,
    ) -> float:
        for _ in range(num_warmup):
            session.run(None, feed_dict)

        start_time = time.perf_counter()

        for _ in range(num_runs):
            session.run(None, feed_dict)

        end_time = time.perf_counter()

        avg_time_ms = ((end_time - start_time) / num_runs) * 1000
        return round(avg_time_ms, 4)

    def run_benchmarks(self):
        results = []

        for model_name, model_path in self.onnx_model_paths.items():
            session = self.load_onnx_session(model_name)
            dummy_input = self.create_dummy_input(batch_size=1)
            dummy_context = self.create_dummy_context(batch_size=1, context_dim=6)
            feed_dict = self.create_feed_dict(
                session=session,
                dummy_input=dummy_input,
                dummy_context=dummy_context,
            )

            model_size_mb = self.caculate_model_size(model_path)
            peak_memory_mb = self.calculate_memory_peak_in_cpu(
                session=session,
                feed_dict=feed_dict,
            )
            inference_time_ms = self.caculate_inference_time_cpu(
                session=session,
                feed_dict=feed_dict,
                num_runs=100,
                num_warmup=10,
            )

            row = {
                "model_name": model_name,
                "model_size_mb": round(model_size_mb, 4),
                "cpu_peak_memory_mb": peak_memory_mb,
                "cpu_inference_time_ms": inference_time_ms,
            }
            results.append(row)

            print(
                f"{model_name}: "
                f"size={row['model_size_mb']} MB, "
                f"peak_memory={row['cpu_peak_memory_mb']} MB, "
                f"inference_time={row['cpu_inference_time_ms']} ms"
            )

        self.results = results
        return self.export_to_DataFrame()

    def export_to_DataFrame(self):
        return pd.DataFrame(
            self.results,
            columns=[
                "model_name",
                "model_size_mb",
                "cpu_peak_memory_mb",
                "cpu_inference_time_ms",
            ],
        )

    def export_to_csv(self, df: pd.DataFrame, filename: str, export_dir: Path):
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        output_path = export_dir / filename
        df.to_csv(output_path, index=False)
        return output_path


if __name__ == "__main__":
    benchmark = EdgeBenchmark(ONNX_MODEL_PATHS)
    dataframe = benchmark.run_benchmarks()
    output_file = benchmark.export_to_csv(
        dataframe,
        filename="edge_benchmark_onnx_results_on_pi.csv",
        export_dir=Path("/home/icnlab/Desktop/moe/results"),
    )
    print(f"Saved benchmark results to {output_file}")
