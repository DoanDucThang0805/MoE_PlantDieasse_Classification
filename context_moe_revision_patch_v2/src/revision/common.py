from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

SEEDS = (42, 43, 44, 45, 46)


def get_dataset_builder(name: str):
    name = name.lower()
    if name in {"plantdoc", "plant_doc"}:
        from dataset.plantdoc_dataset import build_datasets
        return build_datasets, "plantdoc"
    if name in {"slif", "slif_tomato", "slif_tomato_dataset_phase1"}:
        from dataset.slif_tomato_dataset import build_datasets
        return build_datasets, "slif_tomato_dataset_phase1"
    raise ValueError(f"Unsupported dataset: {name}")


def make_loader(dataset: str, split: str = "test", batch_size: int = 32,
                use_context: bool = True, num_workers: int = 0):
    builder, canonical = get_dataset_builder(dataset)
    train_ds, val_ds, test_ds = builder(use_context=use_context)
    ds = {"train": train_ds, "validation": val_ds, "test": test_ds}[split]
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available()), ds, canonical


def find_checkpoint(root: Path, seed: int) -> Path:
    seed_dir = Path(root) / f"seed_{seed}"
    if not seed_dir.exists():
        raise FileNotFoundError(f"Missing seed directory: {seed_dir}")
    runs = sorted(p for p in seed_dir.glob("run_*") if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"No run_* directory under {seed_dir}")
    ckpt = runs[-1] / "best_checkpoint.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing best checkpoint: {ckpt}")
    return ckpt


def unwrap_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def evaluate_predictions(y_true, y_pred) -> Tuple[float, float]:
    return (
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average="macro", zero_division=0),
    )


def count_state_dict_parameters(state_dict: Dict[str, torch.Tensor]) -> int:
    # State dict also contains buffers. Restrict to floating trainable-shaped tensors
    # only when no model instance is available; exact nn.Parameter count should be
    # obtained from an instantiated model. This helper is used mainly for audit.
    return int(sum(v.numel() for v in state_dict.values() if torch.is_tensor(v) and v.is_floating_point()))
