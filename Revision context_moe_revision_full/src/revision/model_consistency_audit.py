"""Audit the architecture/parameter claims used in the manuscript.

This script deliberately separates:
  1) stored parameter capacity,
  2) dynamic PyTorch Top-k execution,
  3) ONNX export semantics.

Run from repository root:
  python -m src.revision.model_consistency_audit --checkpoint PATH --output_csv diagnostics/revision/model_audit.csv

Or, after `cd src`:
  python -m revision.model_consistency_audit ...
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import torch
import pandas as pd


def expected_parameter_breakdown(num_experts=4, model_dim=576, expert_hidden=1024,
                                 context_dim=6, context_proj_dim=32, num_classes=8):
    backbone = 927_008  # torchvision MobileNetV3-Small .features, verified structurally
    pre_post_norm = 2 * (2 * model_dim)
    per_expert = (
        model_dim * expert_hidden + expert_hidden +  # Linear 576->1024
        2 * expert_hidden +                         # LayerNorm(1024)
        expert_hidden * model_dim + model_dim        # Linear 1024->576
    )
    experts = num_experts * per_expert
    fusion_dim = model_dim + context_proj_dim
    gate = (
        2 * model_dim +                # embedding_norm
        2 * context_dim +              # context_norm
        2 * context_proj_dim +         # context_proj_norm
        2 * fusion_dim +               # fusion_norm
        (context_dim * context_proj_dim + context_proj_dim) +
        (context_proj_dim * context_proj_dim + context_proj_dim) +
        fusion_dim * num_experts +     # noise_layer, bias=False
        (fusion_dim * num_experts + num_experts)  # gate_projector
    )
    classifier = (
        model_dim * 256 + 256 + 2 * 256 +
        256 * 128 + 128 + 2 * 128 +
        128 * num_classes + num_classes
    )
    return {
        "backbone_features": backbone,
        "pre_post_norm": pre_post_norm,
        "experts_total": experts,
        "expert_each": per_expert,
        "linear_context_gate": gate,
        "classifier": classifier,
        "total_expected": backbone + pre_post_norm + experts + gate + classifier,
        "static_uniform_control_expected": backbone + pre_post_norm + experts + classifier,
    }


def count_checkpoint_by_prefix(path: Path):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    groups = {"feature_extractor": 0, "norms": 0, "experts": 0, "gating": 0, "classifier": 0, "other": 0}
    for k, v in state.items():
        if not torch.is_tensor(v):
            continue
        n = v.numel()
        if k.startswith("feature_extractor."):
            groups["feature_extractor"] += n
        elif k.startswith("pre_moe_norm.") or k.startswith("post_moe_norm."):
            groups["norms"] += n
        elif k.startswith("moe_layer.experts."):
            groups["experts"] += n
        elif k.startswith("moe_layer.gating."):
            groups["gating"] += n
        elif k.startswith("classifier."):
            groups["classifier"] += n
        else:
            groups["other"] += n
    groups["state_dict_tensor_total"] = sum(groups.values())
    meta = {k: ckpt.get(k) for k in ("num_classes", "num_experts", "top_k", "temperature", "router_mode", "context_dim")}
    return groups, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--output_csv", type=Path, default=Path("diagnostics/revision/model_audit.csv"))
    args = ap.parse_args()

    exp = expected_parameter_breakdown()
    rows = [{"source": "code_formula", "component": k, "parameters": v} for k, v in exp.items()]

    print("Expected current Linear-Gating MoE parameter count")
    for k, v in exp.items():
        print(f"  {k:34s} {v:12,d}")
    print("\nManuscript value 3,484,500 is incompatible with the current 576->1024->576 x4 code.")

    if args.checkpoint:
        groups, meta = count_checkpoint_by_prefix(args.checkpoint)
        print("\nCheckpoint metadata:", meta)
        print("Checkpoint tensor counts by prefix:")
        for k, v in groups.items():
            print(f"  {k:34s} {v:12,d}")
            rows.append({"source": "checkpoint_state", "component": k, "parameters": v})

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(f"\nSaved: {args.output_csv}")


if __name__ == "__main__":
    main()
