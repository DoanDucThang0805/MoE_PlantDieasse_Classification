"""
Benchmark edge-side inference cost for MobileNetV3-Small variants.

The benchmark uses batch size 1 and reports params, parameter memory, FLOPs,
latency, and peak memory. By default it builds architectures with random
weights to avoid downloading pretrained checkpoints; the measured compute cost
does not depend on trained weight values.

Examples:
    python src/benchmark/edge_runtime_memory.py --threads-list 1 4 --runs 200 --warmup 50
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

try:
    import timm
except ImportError:  # pragma: no cover - optional at runtime
    timm = None

try:
    from thop import profile
except ImportError:  # pragma: no cover - optional at runtime
    profile = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


MODEL_NAMES = (
    "mobilenetv3_small",
    "widened_mlp_head",
    "dense_multibranch",
    "mobilenetv3_small_moe",
)


class DenseBranch(nn.Module):
    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class MobileNetV3SmallDenseMultiBranchNoWeights(nn.Module):
    """Equivalent to the repo dense multi-branch model, without weight download."""

    def __init__(self, num_classes: int = 8, num_experts: int = 4) -> None:
        super().__init__()
        backbone = mobilenet_v3_small(weights=None)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.flatten = nn.Flatten(1)
        in_features = backbone.classifier[0].in_features
        self.experts = nn.ModuleList(
            [DenseBranch(in_features, num_classes) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return torch.stack([expert(x) for expert in self.experts], dim=0).mean(dim=0)


class MobileNetV3SmallFeatureExtractorNoWeights(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = mobilenet_v3_small(weights=None)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output_dim = 576

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


def make_baseline(num_classes: int, baseline_source: str) -> nn.Module:
    if baseline_source == "timm":
        if timm is None:
            raise ImportError("timm is required for --baseline-source timm")
        return timm.create_model(
            "mobilenetv3_small_100",
            pretrained=False,
            num_classes=num_classes,
        )
    return mobilenet_v3_small(weights=None, num_classes=num_classes)


def make_widened_head(num_classes: int) -> nn.Module:
    model = mobilenet_v3_small(weights=None)
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 2048),
        nn.Hardswish(),
        nn.Dropout(0.2),
        nn.Linear(2048, 512),
        nn.Hardswish(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )
    return model


def make_moe(
    num_classes: int,
    context_dim: int,
    num_experts: int,
    top_k: int,
    temperature: float,
) -> nn.Module:
    from models.moe.model import MoEModel

    model = MoEModel(
        context_dim=context_dim,
        num_classes=num_classes,
        num_experts=num_experts,
        top_k=top_k,
        router_mode="context_aware",
        temperature=temperature,
    )
    model.feature_extractor = MobileNetV3SmallFeatureExtractorNoWeights()
    return model


def build_model(args: argparse.Namespace) -> Tuple[nn.Module, Tuple[torch.Tensor, ...]]:
    image = torch.randn(1, 3, args.image_size, args.image_size)

    if args.model == "mobilenetv3_small":
        return make_baseline(args.num_classes, args.baseline_source), (image,)
    if args.model == "widened_mlp_head":
        return make_widened_head(args.num_classes), (image,)
    if args.model == "dense_multibranch":
        model = MobileNetV3SmallDenseMultiBranchNoWeights(
            num_classes=args.num_classes,
            num_experts=args.dense_experts,
        )
        return model, (image,)
    if args.model == "mobilenetv3_small_moe":
        context = torch.randn(1, args.context_dim)
        model = make_moe(
            num_classes=args.num_classes,
            context_dim=args.context_dim,
            num_experts=args.moe_experts,
            top_k=args.top_k,
            temperature=args.temperature,
        )
        return model, (image, context)
    raise ValueError(f"Unsupported model: {args.model}")


def parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def state_dict_size_mb(model: nn.Module) -> float:
    return sum(tensor.numel() * tensor.element_size() for tensor in model.state_dict().values()) / 1e6


def serialized_size_mb(model: nn.Module) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as handle:
        path = Path(handle.name)
    try:
        torch.save(model.state_dict(), path)
        return path.stat().st_size / 1e6
    finally:
        path.unlink(missing_ok=True)


def current_rss_mb() -> float:
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / 1024.0 if sys.platform != "darwin" else rss / (1024.0 * 1024.0)
    except Exception:
        return float("nan")


class RssSampler:
    def __init__(self, interval_s: float = 0.001) -> None:
        self.interval_s = interval_s
        self.peak_mb = current_rss_mb()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, current_rss_mb())
            time.sleep(self.interval_s)

    def __enter__(self) -> "RssSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_mb = max(self.peak_mb, current_rss_mb())


def synchronize(device: torch.device) -> None:
    return None


@torch.inference_mode()
def benchmark_latency_and_memory(
    model: nn.Module,
    inputs: Tuple[torch.Tensor, ...],
    device: torch.device,
    warmup: int,
    runs: int,
) -> Tuple[Dict[str, float], float]:
    model.to(device).eval()
    inputs = tuple(tensor.to(device) for tensor in inputs)

    for _ in range(warmup):
        model(*inputs)
    synchronize(device)

    peak_context = RssSampler()

    timings_ms: List[float] = []
    context_manager = peak_context if peak_context is not None else _NullContext()
    with context_manager:
        for _ in range(runs):
            start = time.perf_counter()
            model(*inputs)
            synchronize(device)
            timings_ms.append((time.perf_counter() - start) * 1000.0)

    peak_memory_mb = float(peak_context.peak_mb)

    timings_ms.sort()
    latency = {
        "latency_mean_ms": statistics.fmean(timings_ms),
        "latency_median_ms": statistics.median(timings_ms),
        "latency_p95_ms": timings_ms[min(len(timings_ms) - 1, int(0.95 * len(timings_ms)))],
    }
    return latency, peak_memory_mb


class _NullContext:
    def __enter__(self) -> "_NullContext":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def compute_flops(
    model: nn.Module,
    inputs: Tuple[torch.Tensor, ...],
    device: torch.device,
) -> Optional[float]:
    if profile is None:
        return None
    model.to(device).eval()
    inputs = tuple(tensor.to(device) for tensor in inputs)
    with torch.inference_mode():
        flops, _ = profile(model, inputs=inputs, verbose=False)
    return float(flops)


def hardware_label(device: torch.device) -> str:
    cpu_name = platform.processor() or platform.machine()
    return f"{cpu_name} CPU, {os.cpu_count()} logical cores"


def benchmark_one(args: argparse.Namespace) -> Dict[str, object]:
    if args.threads is not None:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)

    device = torch.device(args.device)
    if device.type != "cpu":
        raise ValueError("This benchmark is CPU-only. Use --device cpu.")
    model, inputs = build_model(args)

    params = parameter_count(model)
    flops = compute_flops(model, inputs, device)
    latency, peak_memory_mb = benchmark_latency_and_memory(
        model=model,
        inputs=inputs,
        device=device,
        warmup=args.warmup,
        runs=args.runs,
    )

    return {
        "model": args.model,
        "params_m": params / 1e6,
        "model_size_mb": state_dict_size_mb(model),
        "serialized_state_dict_mb": serialized_size_mb(model),
        "flops_g": None if flops is None else flops / 1e9,
        "latency_ms_img": latency["latency_median_ms"],
        "latency_mean_ms_img": latency["latency_mean_ms"],
        "latency_p95_ms_img": latency["latency_p95_ms"],
        "peak_memory_mb": peak_memory_mb,
        "batch_size": 1,
        "device": str(device),
        "hardware": hardware_label(device),
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "runs": args.runs,
        "warmup": args.warmup,
    }


def run_child_for_model(args: argparse.Namespace, model_name: str, threads: int) -> Dict[str, object]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        model_name,
        "--device",
        args.device,
        "--num-classes",
        str(args.num_classes),
        "--image-size",
        str(args.image_size),
        "--context-dim",
        str(args.context_dim),
        "--moe-experts",
        str(args.moe_experts),
        "--top-k",
        str(args.top_k),
        "--temperature",
        str(args.temperature),
        "--dense-experts",
        str(args.dense_experts),
        "--baseline-source",
        args.baseline_source,
        "--warmup",
        str(args.warmup),
        "--runs",
        str(args.runs),
        "--json",
        "--threads",
        str(threads),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def write_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: object, digits: int) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_markdown(rows: Sequence[Dict[str, object]]) -> None:
    print("| Threads | Model | Params (M) | Model size (MB) | FLOPs (G) | Latency (ms/img) | Peak memory (MB) |")
    print("|---:|---|---:|---:|---:|---:|---:|")
    labels = {
        "mobilenetv3_small": "MobileNetV3-Small",
        "widened_mlp_head": "MobileNetV3-Small + widened MLP head",
        "dense_multibranch": "MobileNetV3-Small + dense multi-branch head",
        "mobilenetv3_small_moe": "MobileNetV3-Small-MoE",
    }
    for row in rows:
        print(
            "| "
            + " | ".join(
                [
                    format_value(row["threads"], 0),
                    labels[str(row["model"])],
                    format_value(row["params_m"], 3),
                    format_value(row["model_size_mb"], 2),
                    format_value(row["flops_g"], 3),
                    format_value(row["latency_ms_img"], 2),
                    format_value(row["peak_memory_mb"], 2),
                ]
            )
            + " |"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*MODEL_NAMES, "all"), default="all")
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--threads-list",
        type=int,
        nargs="+",
        default=[1, 4],
        help="CPU thread counts to benchmark when --model all is used.",
    )
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--context-dim", type=int, default=6)
    parser.add_argument("--moe-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--dense-experts", type=int, default=4)
    parser.add_argument("--baseline-source", choices=("timm", "torchvision"), default="timm")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "diagnostics" / "edge_runtime_memory.csv")
    parser.add_argument("--json", action="store_true", help="Print a single JSON result for subprocess use.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model != "all":
        if args.json:
            if args.threads is None:
                raise ValueError("--json requires --threads.")
            with contextlib.redirect_stdout(io.StringIO()):
                rows = [benchmark_one(args)]
            print(json.dumps(rows[0]))
        else:
            if args.threads is not None:
                rows = [benchmark_one(args)]
            else:
                rows = [run_child_for_model(args, args.model, threads) for threads in args.threads_list]
            print_markdown(rows)
            print(f"\nHardware: {rows[0]['hardware']}")
        return

    rows = [
        run_child_for_model(args, model_name, threads)
        for threads in args.threads_list
        for model_name in MODEL_NAMES
    ]
    write_csv(rows, args.output)
    print_markdown(rows)
    print(f"\nHardware: {rows[0]['hardware']}")
    print(f"CSV: {args.output}")


if __name__ == "__main__":
    main()
