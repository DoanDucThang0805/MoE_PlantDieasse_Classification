import argparse
import sys
from pathlib import Path
from typing import Dict

import torch


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.moe.modelv2 import MoEModel


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict")
        if state_dict is None:
            state_dict = checkpoint.get("state_dict")
        if state_dict is None:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)


def export_mobilenetv3small_moe_to_onnx(
    output_path: Path,
    checkpoint_path: Path | None = None,
    num_classes: int = 8,
    num_experts: int = 4,
    top_k: int = 2,
    context_dim: int = 6,
    router_mode: str = "context_aware",
    temperature: float = 0.5,
    opset_version: int = 18,
) -> Path:
    model = MoEModel(
        context_dim=context_dim,
        num_classes=num_classes,
        num_experts=num_experts,
        top_k=top_k,
        router_mode=router_mode,
        temperature=temperature,
    )

    if checkpoint_path is not None:
        load_checkpoint(model, checkpoint_path)
        print(f"Loaded checkpoint: {checkpoint_path}")

    model.eval()

    image = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    context = torch.randn(1, context_dim, dtype=torch.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes: Dict[str, Dict[int, str]] = {
        "image": {0: "batch_size"},
        "context": {0: "batch_size"},
        "logits": {0: "batch_size"},
        "router_logits": {0: "batch_size"},
        "top_k_indices": {0: "batch_size"},
    }

    torch.onnx.export(
        model,
        (image, context),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["image", "context"],
        output_names=["logits", "router_logits", "top_k_indices"],
        dynamic_axes=dynamic_axes,
    )

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ONNX-friendly MobileNetV3Small MoE model."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to best_checkpoint.pth or last_checkpoint.pth.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("onnx/mobilenetv3small_moe.onnx"),
        help="Output ONNX file path.",
    )
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--context_dim", type=int, default=6)
    parser.add_argument(
        "--router_mode",
        type=str,
        default="context_aware",
        choices=["context_aware", "noisy"],
    )
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--opset_version", type=int, default=18)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = export_mobilenetv3small_moe_to_onnx(
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        num_classes=args.num_classes,
        num_experts=args.num_experts,
        top_k=args.top_k,
        context_dim=args.context_dim,
        router_mode=args.router_mode,
        temperature=args.temperature,
        opset_version=args.opset_version,
    )
    print(f"Exported ONNX model to: {output_path}")


if __name__ == "__main__":
    main()
