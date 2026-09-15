"""
Seed-Wise Paired Statistical Test.

Computes paired t-test and Wilcoxon signed-rank test between:
  - PRIMARY: MobileNetV3-Small-MoE vs MobileNetV3-Small (PlantDoc & SLIF-Tomato)
  - EXPLORATORY: MoE vs external backbones with Holm-Bonferroni & BH correction

Output: paired_statistics.csv with columns:
  dataset, metric, model_A, model_B, n_seeds,
  mean_A, std_A, mean_B, std_B, mean_delta,
  paired_t_p, wilcoxon_p, holm_p, bh_p, conclusion
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
from scipy.stats import ttest_rel, wilcoxon
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------------------------------------------------------
# Path setup — must happen before any local imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ALL pretrained model .py files call torchinfo.summary() at module level,
# which prints noise and is slow.  We build every model via clean factory
# functions (timm / torchvision) to avoid those side-effects entirely.
import timm
import copy
import torch.nn as nn

from models.moe.linear_model import MoEModel as LinearGatingMoE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEEDS = [42, 43, 44, 45, 46]
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_PLANTDOC = "/media/data/minhht/context_moe/checkpoints/plantdoc"
BASE_SLIF = "/media/data/minhht/context_moe/checkpoints/slif_tomato_dataset_phase1"


# ---------------------------------------------------------------------------
# Dataset configs (built lazily to avoid side-effects at import time)
# ---------------------------------------------------------------------------

def get_test_loader(dataset_name: str):
    """Return a DataLoader for the test split of the given dataset."""
    if dataset_name == "plantdoc":
        from dataset.plantdoc_dataset import build_datasets
    elif dataset_name == "slif":
        from dataset.slif_tomato_dataset import build_datasets
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    _, _, test_ds = build_datasets(use_context=True)
    return DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=4, pin_memory=True)


# ---------------------------------------------------------------------------
# Model factory — clean construction without module-level side-effects
#
# Every function below mirrors the architecture of the corresponding file
# in models/pretrained_model/*.py, but does NOT call torchinfo.summary().
# Weights are loaded from seed checkpoints, so pretrained=False for timm
# models and weights=None for torchvision models.
# ---------------------------------------------------------------------------

def make_moe():
    """Linear-Gated MoE on MobileNetV3-Small (the proposed model)."""
    return LinearGatingMoE(
        context_dim=6, num_classes=8, num_experts=4,
        top_k=2, router_mode="context_aware", temperature=0.5,
    )


def make_mobilenetv3_small():
    """MobileNetV3-Small — mirrors mobilenetv3_small.py."""
    return timm.create_model(
        "mobilenetv3_small_100.lamb_in1k",
        pretrained=False,
        num_classes=8,
    )


def make_efficientnetb0():
    """EfficientNet-B0 — mirrors efficientnetb0.py."""
    return timm.create_model(
        "efficientnet_b0.ra_in1k",
        pretrained=False,
        num_classes=8,
    )


def make_ghostnet():
    """GhostNet — mirrors ghostnet.py."""
    return timm.create_model(
        "ghostnet_100",
        pretrained=False,
        num_classes=8,
    )


def make_shufflenetv2():
    """ShuffleNetV2 x2.0 — mirrors shufflenetv2.py (torchvision)."""
    from torchvision.models import shufflenet_v2_x2_0
    model = shufflenet_v2_x2_0(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 8)
    return model


def make_squeezenet():
    """SqueezeNet 1.1 — mirrors squeezenet.py (torchvision)."""
    from torchvision.models import squeezenet1_1
    model = squeezenet1_1(weights=None)
    model.classifier[1] = nn.Conv2d(512, 8, kernel_size=1)
    model.num_classes = 8
    return model


def make_mobilevit_s():
    """MobileViT-S — mirrors mobilevits.py."""
    return timm.create_model(
        "mobilevit_s.cvnets_in1k",
        pretrained=False,
        num_classes=8,
    )


MODEL_FACTORY = {
    "MobileNetV3-Small-MoE": (make_moe, True),
    "MobileNetV3-Small":     (make_mobilenetv3_small, False),
    "EfficientNetB0":        (make_efficientnetb0, False),
    "GhostNet":              (make_ghostnet, False),
    "ShuffleNetV2":          (make_shufflenetv2, False),
    "SqueezeNet":            (make_squeezenet, False),
    "MobileViT-S":           (make_mobilevit_s, False),
}

# checkpoint sub-paths relative to BASE_<dataset>
CKPT_SUBPATH = {
    "plantdoc": {
        "MobileNetV3-Small-MoE":
            "moe_linearcontextaware_temp0.5/mobilenetv3small_moe/4_experts/top_2",
        "MobileNetV3-Small": "pretrain_models/mobilenetv3_small",
        "EfficientNetB0":    "pretrain_models/efficientnetb0",
        "GhostNet":          "pretrain_models/ghostnet",
        "ShuffleNetV2":      "pretrain_models/shufflenetv2",
        "SqueezeNet":        "pretrain_models/squeezenet",
        "MobileViT-S":       "pretrain_models/mobilevits",
    },
    "slif": {
        "MobileNetV3-Small-MoE":
            "moe_linearcontextaware_temp0.5/mobilenetv3small_moe/4_experts/top_2",
        "MobileNetV3-Small": "pretrain_models/mobilenetv3_small",
        "EfficientNetB0":    "pretrain_models/efficientnetb0",
        "GhostNet":          "pretrain_models/ghostnet",
        "ShuffleNetV2":      "pretrain_models/shufflenetv2",
        "SqueezeNet":        "pretrain_models/squeezenet",
        "MobileViT-S":       "pretrain_models/mobilevits",
    },
}

BASE_MAP = {"plantdoc": BASE_PLANTDOC, "slif": BASE_SLIF}


# ---------------------------------------------------------------------------
# Checkpoint & inference helpers
# ---------------------------------------------------------------------------

def find_checkpoint(root: str, seed: int) -> str:
    """Locate best_checkpoint.pth for a given seed under root."""
    seed_dir = os.path.join(root, f"seed_{seed}")
    if not os.path.exists(seed_dir):
        raise FileNotFoundError(f"Seed dir not found: {seed_dir}")
    runs = sorted(
        d for d in os.listdir(seed_dir)
        if os.path.isdir(os.path.join(seed_dir, d)) and d.startswith("run_")
    )
    if not runs:
        raise FileNotFoundError(f"No run_* dirs in {seed_dir}")
    if len(runs) > 1:
        print(f"  [WARN] Multiple runs in seed_{seed}, using latest: {runs[-1]}")
    ckpt = os.path.join(seed_dir, runs[-1], "best_checkpoint.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return ckpt


def load_model(model_fn, ckpt_path: str):
    """Instantiate a model via its factory, then load checkpoint weights."""
    model = model_fn()
    state = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in state:
                state = state[key]
                break
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


@torch.no_grad()
def run_inference(model, loader, is_moe: bool):
    """Run inference and return (accuracy, macro-F1).

    Apply softmax before argmax to match the behaviour in
    paired_checkpoint_test.py (softmax is monotone so prediction is identical,
    but keeping it consistent avoids confusion if the code is later extended).
    """
    preds_all, labels_all = [], []
    for images, labels, contexts in loader:
        images = images.to(DEVICE)
        contexts = contexts.to(DEVICE)
        if is_moe:
            logits, _, _ = model(images, contexts)
        else:
            logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        preds_all.append(preds.cpu().numpy())
        labels_all.append(labels.cpu().numpy())
    y_pred = np.concatenate(preds_all)
    y_true = np.concatenate(labels_all)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return acc, f1


# Cache DataLoader per dataset — test split is deterministic
# (fixed random_state=42 in LoadDataset), so one loader suffices.
_loader_cache: dict = {}


def get_cached_loader(dataset: str) -> DataLoader:
    """Return (and cache) the test DataLoader for the given dataset."""
    if dataset not in _loader_cache:
        _loader_cache[dataset] = get_test_loader(dataset)
    return _loader_cache[dataset]


def collect_seed_metrics(model_name: str, dataset: str) -> dict:
    """Run inference for all seeds, return {acc: array(5,), f1: array(5,)}."""
    fn, is_moe = MODEL_FACTORY[model_name]
    base = BASE_MAP[dataset]
    subpath = CKPT_SUBPATH[dataset][model_name]
    ckpt_root = os.path.join(base, subpath)
    loader = get_cached_loader(dataset)

    accs, f1s = [], []
    for seed in SEEDS:
        ckpt = find_checkpoint(ckpt_root, seed)
        model = load_model(fn, ckpt)
        acc, f1 = run_inference(model, loader, is_moe)
        accs.append(acc)
        f1s.append(f1)
        print(f"    [seed {seed}] Acc={acc:.4f}  F1={f1:.4f}")
        del model
        torch.cuda.empty_cache()

    return {"acc": np.array(accs), "f1": np.array(f1s)}


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def paired_tests(a: np.ndarray, b: np.ndarray):
    """Return (mean_delta, p_ttest, p_wilcoxon)."""
    delta = a - b
    mean_delta = delta.mean()
    _, p_t = ttest_rel(a, b)
    try:
        _, p_w = wilcoxon(delta)
    except ValueError:
        p_w = float("nan")
    return mean_delta, p_t, p_w


def holm_bonferroni(p_values: list) -> list:
    """Holm-Bonferroni correction. Returns adjusted p-values."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [None] * n
    prev_adj = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = max(prev_adj, p * (n - rank))
        adj = min(adj, 1.0)
        adjusted[orig_idx] = adj
        prev_adj = adj
    return adjusted


def benjamini_hochberg(p_values: list) -> list:
    """
    Benjamini-Hochberg FDR correction (stepdown).
    Sorts p-values ascending, then applies stepdown from largest rank.
    Returns adjusted p-values in the original order.
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [None] * n
    prev_adj = 1.0
    for step in range(n - 1, -1, -1):
        orig_idx, p = indexed[step]
        rank_1indexed = step + 1
        adj = min(prev_adj, p * n / rank_1indexed)
        adj = min(adj, 1.0)
        adjusted[orig_idx] = adj
        prev_adj = adj
    return adjusted


def make_conclusion(mean_delta: float, p_t: float, p_w: float,
                    holm_p: float, bh_p: float, is_primary: bool) -> str:
    """
    Conclude significance level:
      - primary pairs: use paired t-test + Wilcoxon
      - exploratory pairs: use Holm-corrected p
    """
    if is_primary:
        if p_t < 0.05 and p_w < 0.05:
            direction = "A>B" if mean_delta > 0 else "A<B"
            return f"significant ({direction}, both tests)"
        elif p_t < 0.05:
            direction = "A>B" if mean_delta > 0 else "A<B"
            return f"partially_supported ({direction}, t-test only)"
        else:
            return "not_significant"
    else:
        if holm_p < 0.05:
            direction = "A>B" if mean_delta > 0 else "A<B"
            return f"exploratory_significant ({direction}, Holm p={holm_p:.4f})"
        else:
            return f"exploratory_not_significant (Holm p={holm_p:.4f})"


# ---------------------------------------------------------------------------
# Comparison plan
# ---------------------------------------------------------------------------

# PRIMARY comparisons (MoE vs baseline — the core claim of the paper)
PRIMARY = [
    ("plantdoc", "MobileNetV3-Small-MoE", "MobileNetV3-Small", True),
    ("slif",     "MobileNetV3-Small-MoE", "MobileNetV3-Small", True),
]

# EXPLORATORY comparisons (external reference backbones)
# PlantDoc: all lightweight baselines
# SLIF-Tomato: only models with checkpoints on that dataset
EXPLORATORY = [
    # PlantDoc
    ("plantdoc", "MobileNetV3-Small-MoE", "EfficientNetB0",  False),
    ("plantdoc", "MobileNetV3-Small-MoE", "GhostNet",        False),
    ("plantdoc", "MobileNetV3-Small-MoE", "ShuffleNetV2",    False),
    ("plantdoc", "MobileNetV3-Small-MoE", "SqueezeNet",      False),
    ("plantdoc", "MobileNetV3-Small-MoE", "MobileViT-S",     False),
    # SLIF-Tomato
    ("slif",     "MobileNetV3-Small-MoE", "EfficientNetB0",  False),
    ("slif",     "MobileNetV3-Small-MoE", "GhostNet",        False),
    ("slif",     "MobileNetV3-Small-MoE", "ShuffleNetV2",    False),
    ("slif",     "MobileNetV3-Small-MoE", "SqueezeNet",      False),
    ("slif",     "MobileNetV3-Small-MoE", "MobileViT-S",     False),
]

ALL_COMPARISONS = PRIMARY + EXPLORATORY


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(output_csv: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    # ── Step 1: collect per-seed metrics for every unique (dataset, model) ──
    needed = set()
    for dataset, model_a, model_b, _ in ALL_COMPARISONS:
        needed.add((dataset, model_a))
        needed.add((dataset, model_b))

    cache = {}
    for dataset, model_name in sorted(needed):
        key = (dataset, model_name)
        if key not in cache:
            print(f"\n▶ [{dataset}] {model_name}")
            # Skip if checkpoint subpath not configured for this dataset
            if model_name not in CKPT_SUBPATH.get(dataset, {}):
                print(f"  [SKIP] No checkpoint config for {model_name} on {dataset}")
                cache[key] = None
                continue
            try:
                cache[key] = collect_seed_metrics(model_name, dataset)
            except FileNotFoundError as e:
                print(f"  [SKIP] {e}")
                cache[key] = None

    # ── Step 2: gather p-values for MTC ─────────────────────────────────────
    # We collect ALL (dataset, modelA, modelB, metric) p-values for Holm/BH
    rows_raw = []  # list of dict (unpopulated holm_p, bh_p)

    for dataset, model_a, model_b, is_primary in ALL_COMPARISONS:
        data_a = cache.get((dataset, model_a))
        data_b = cache.get((dataset, model_b))
        if data_a is None or data_b is None:
            print(f"\n[WARN] Skipping {model_a} vs {model_b} on {dataset}: missing data")
            continue

        for metric_key, metric_label in [("acc", "Accuracy"), ("f1", "Macro-F1")]:
            a = data_a[metric_key]
            b = data_b[metric_key]
            mean_delta, p_t, p_w = paired_tests(a, b)

            rows_raw.append({
                "dataset":    dataset,
                "metric":     metric_label,
                "model_A":    model_a,
                "model_B":    model_b,
                "n_seeds":    len(SEEDS),
                "mean_A":     float(a.mean()),
                "std_A":      float(a.std(ddof=1)),
                "mean_B":     float(b.mean()),
                "std_B":      float(b.std(ddof=1)),
                "mean_delta": float(mean_delta),
                "paired_t_p": float(p_t),
                "wilcoxon_p": float(p_w) if not np.isnan(p_w) else float("nan"),
                "is_primary": is_primary,
                # placeholders
                "holm_p": None,
                "bh_p":   None,
                "conclusion": None,
            })

    # ── Step 3: apply MTC only to exploratory rows ──────────────────────────
    exp_indices = [i for i, r in enumerate(rows_raw) if not r["is_primary"]]
    exp_p_t = [rows_raw[i]["paired_t_p"] for i in exp_indices]

    if exp_p_t:
        holm_adj = holm_bonferroni(exp_p_t)
        bh_adj   = benjamini_hochberg(exp_p_t)
        for rank, i in enumerate(exp_indices):
            rows_raw[i]["holm_p"] = holm_adj[rank]
            rows_raw[i]["bh_p"]   = bh_adj[rank]

    # Primary rows: set holm_p = bh_p = paired_t_p (no correction needed)
    for i, r in enumerate(rows_raw):
        if r["is_primary"]:
            rows_raw[i]["holm_p"] = r["paired_t_p"]
            rows_raw[i]["bh_p"]   = r["paired_t_p"]

    # ── Step 4: determine conclusion ────────────────────────────────────────
    for r in rows_raw:
        r["conclusion"] = make_conclusion(
            r["mean_delta"], r["paired_t_p"], r["wilcoxon_p"],
            r["holm_p"], r["bh_p"], r["is_primary"],
        )

    # ── Step 5: write CSV ────────────────────────────────────────────────────
    fieldnames = [
        "dataset", "metric", "model_A", "model_B", "n_seeds",
        "mean_A", "std_A", "mean_B", "std_B", "mean_delta",
        "paired_t_p", "wilcoxon_p", "holm_p", "bh_p", "conclusion",
    ]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows_raw:
            row_out = {k: (f"{r[k]:.6f}" if isinstance(r[k], float) and not np.isnan(r[k])
                           else ("nan" if isinstance(r[k], float) else r[k]))
                       for k in fieldnames}
            writer.writerow(row_out)

    # ── Step 6: print summary table ──────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"{'Dataset':<10} {'Metric':<10} {'Model A':<28} {'Model B':<24} "
          f"{'Δ̄':>8} {'t-p':>7} {'W-p':>7} {'Holm-p':>8} {'BH-p':>7}  Conclusion")
    print("-" * 90)
    for r in rows_raw:
        tag = "[PRI]" if r["is_primary"] else "[EXP]"
        p_w_str = f"{r['wilcoxon_p']:7.4f}" if not np.isnan(r['wilcoxon_p']) else "    nan"
        print(f"{r['dataset']:<10} {r['metric']:<10} {r['model_A']:<28} {r['model_B']:<24} "
              f"{r['mean_delta']:+8.4f} {r['paired_t_p']:7.4f} {p_w_str} "
              f"{r['holm_p']:8.4f} {r['bh_p']:7.4f}  {tag} {r['conclusion']}")
    print("=" * 90)
    print(f"\n✔ Results saved to: {output_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Seed-wise paired statistical test (MoE vs baseline)."
    )
    parser.add_argument(
        "--output_csv",
        default="/media/data/minhht/context_moe/diagnostics/statistical_tests/paired_statistics.csv",
        help="Path to output CSV file.",
    )
    args = parser.parse_args()
    main(args.output_csv)
