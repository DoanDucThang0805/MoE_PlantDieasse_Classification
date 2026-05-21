import os
import copy
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import ttest_rel, wilcoxon
from tabulate import tabulate

# =============================================================================
# Imports
# =============================================================================
from dataset.plantdoc_dataset import build_datasets
from models.moe.linear_model import MoEModel as LinearGatingMoE
from models.moe.mlp_model import MoEModel as MLPGatingMoE
from models.dense_multibranch.mobilenetv3_small_dense_multibranch import MobileNetV3SmallDenseMultiBranch
from models.pretrained_model.efficientnetb0 import model as EfficientNetB0
from models.pretrained_model.ghostnet import model as GhostNet
from models.pretrained_model.mobilenetv3_small import model as MobileNetV3Small
from models.pretrained_model.shufflenetv2 import model as ShuffleNetV2
from models.pretrained_model.squeezenet import model as SqueezeNet
from models.pretrained_model.widense_mlp_head import model as WidenedMLPHead

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS       = [42, 43, 44, 45, 46]
BATCH_SIZE  = 32

BASE = "/media/data/minhht/context_moe/checkpoints/plantdoc"

# =============================================================================
# Validation DataLoader
# =============================================================================
_, _, test_dataset = build_datasets(use_context=True)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

# =============================================================================
# Model configs
# Các pretrained model được import sẵn dưới dạng instance nên dùng deepcopy.
# Các MoE model khởi tạo mới mỗi lần qua lambda.
#
# is_moe=True  → logits = model(image, context)
# is_moe=False → logits = model(image)
#
# "MoE" = Linear Gating variant (model chính dùng để so sánh)
# "MLP Gating" = MLP Gating variant (baseline ablation)
# =============================================================================
MODEL_CONFIGS = [
    # ── Main MoE model (Linear context-aware gating) ─────────────────────────
    {
        "name": "Linear Gating",
        "checkpoint_root": f"{BASE}/moe_linearcontextaware_temp0.5/mobilenetv3small_moe/4_experts/top_2",
        "model_fn": lambda: LinearGatingMoE(
            context_dim=6, num_classes=8, num_experts=4,
            top_k=2, router_mode='context_aware', temperature=0.5
        ),
        "is_moe": True,
    },
    # ── Ablation: MLP Gating ──────────────────────────────────────────────────
    {
        "name": "MLP Gating",
        "checkpoint_root": f"{BASE}/moe_contextaware_temp0.5/mobilenetv3small_moe/4_experts/top_2",
        "model_fn": lambda: MLPGatingMoE(
            context_dim=6, num_classes=8, num_experts=4,
            top_k=2, router_mode='context_aware', temperature=0.5
        ),
        "is_moe": True,
    },
    # ── Baselines ─────────────────────────────────────────────────────────────
    {
        "name": "MobileNetV3-Small",
        "checkpoint_root": f"{BASE}/pretrain_models/mobilenetv3_small",
        "model_fn": lambda: copy.deepcopy(MobileNetV3Small),
        "is_moe": False,
    },
    {
        "name": "Widened MLP Head",
        "checkpoint_root": f"{BASE}/pretrain_models/widense_mlp_head",
        "model_fn": lambda: copy.deepcopy(WidenedMLPHead),
        "is_moe": False,
    },
    {
        "name": "Dense Multi-Branch Head",
        "checkpoint_root": f"{BASE}/dense_multibranch/mobilenetv3small_dense_multibranch/4_experts",
        "model_fn": lambda: MobileNetV3SmallDenseMultiBranch(num_classes=8, num_experts=4),
        "is_moe": False,
    },
    # ── Lightweight backbones (MoE vs lightweight, nếu có paired runs) ────────
    {
        "name": "EfficientNetB0",
        "checkpoint_root": f"{BASE}/pretrain_models/efficientnetb0",
        "model_fn": lambda: copy.deepcopy(EfficientNetB0),
        "is_moe": False,
    },
    {
        "name": "GhostNet",
        "checkpoint_root": f"{BASE}/pretrain_models/ghostnet",
        "model_fn": lambda: copy.deepcopy(GhostNet),
        "is_moe": False,
    },
    {
        "name": "ShuffleNetV2",
        "checkpoint_root": f"{BASE}/pretrain_models/shufflenetv2",
        "model_fn": lambda: copy.deepcopy(ShuffleNetV2),
        "is_moe": False,
    },
    {
        "name": "SqueezeNet",
        "checkpoint_root": f"{BASE}/pretrain_models/squeezenet",
        "model_fn": lambda: copy.deepcopy(SqueezeNet),
        "is_moe": False,
    },
]

# =============================================================================
# Các cặp so sánh
# =============================================================================
COMPARISON_PAIRS = [
    # Linear Gating vs tất cả các model còn lại
    ("Linear Gating", "MobileNetV3-Small"),
    ("Linear Gating", "Widened MLP Head"),
    ("Linear Gating", "Dense Multi-Branch Head"),
    ("Linear Gating", "MLP Gating"),
    ("Linear Gating", "EfficientNetB0"),
    ("Linear Gating", "GhostNet"),
    ("Linear Gating", "ShuffleNetV2"),
    ("Linear Gating", "SqueezeNet"),
]

# =============================================================================
# Inference
# =============================================================================

def load_model(model_fn, checkpoint_path):
    model = model_fn()
    state = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in state:
                state = state[key]
                break
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def run_inference(model, loader, is_moe):
    all_preds, all_labels = [], []
    for batch in loader:
        images, labels, contexts = batch
        images   = images.to(DEVICE)
        contexts = contexts.to(DEVICE)
        if is_moe:
            logits, _, _ = model(images, contexts)
        else:
            logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, f1


def find_checkpoint(checkpoint_root, seed):
    seed_dir = os.path.join(checkpoint_root, f"seed_{seed}")
    assert os.path.exists(seed_dir), f"Không tìm thấy: {seed_dir}"

    # Tìm thư mục run_{datetime} bên trong seed_dir
    run_dirs = sorted([
        d for d in os.listdir(seed_dir)
        if os.path.isdir(os.path.join(seed_dir, d)) and d.startswith("run_")
    ])
    assert len(run_dirs) > 0, f"Không tìm thấy thư mục run_* trong: {seed_dir}"
    if len(run_dirs) > 1:
        print(f"  [WARNING] Có {len(run_dirs)} run trong seed_{seed}, lấy run mới nhất: {run_dirs[-1]}")

    ckpt_path = os.path.join(seed_dir, run_dirs[-1], "best_checkpoint.pth")
    assert os.path.exists(ckpt_path), f"Không tìm thấy checkpoint: {ckpt_path}"
    return ckpt_path


def collect_seed_metrics(config):
    accs, f1s = [], []
    for seed in SEEDS:
        ckpt_path = find_checkpoint(config["checkpoint_root"], seed)
        model = load_model(config["model_fn"], ckpt_path)
        acc, f1 = run_inference(model, test_loader, config["is_moe"])
        accs.append(acc)
        f1s.append(f1)
        print(f"  [seed {seed}] Acc={acc:.4f}  F1={f1:.4f}")

    return {"acc": np.array(accs), "f1": np.array(f1s)}


# =============================================================================
# Statistical tests
# =============================================================================

def paired_tests(a, b):
    delta      = a - b
    delta_mean = delta.mean()
    _, p_t     = ttest_rel(a, b)
    try:
        _, p_w = wilcoxon(delta)
    except ValueError:
        p_w = float("nan")
    return delta_mean, p_t, p_w


# =============================================================================
# Main
# =============================================================================

def main():
    print("=== Inference theo seed ===")
    all_metrics = {}
    # Chỉ chạy inference cho các model thực sự xuất hiện trong COMPARISON_PAIRS
    needed = set(m for pair in COMPARISON_PAIRS for m in pair)
    for cfg in MODEL_CONFIGS:
        if cfg["name"] not in needed:
            continue
        # Tránh chạy lại nếu hai entry trỏ cùng checkpoint (MoE = MLP Gating)
        if cfg["name"] in all_metrics:
            continue
        print(f"\n[{cfg['name']}]")
        all_metrics[cfg["name"]] = collect_seed_metrics(cfg)

    print("\n=== Paired Statistical Tests ===")
    rows = []
    for name_a, name_b in COMPARISON_PAIRS:
        for metric_key, metric_label in [("acc", "Accuracy"), ("f1", "mF1")]:
            a = all_metrics[name_a][metric_key]
            b = all_metrics[name_b][metric_key]
            delta_mean, p_t, p_w = paired_tests(a, b)

            rows.append([
                f"{name_a} vs {name_b}",
                metric_label,
                f"{delta_mean:+.4f}",
                f"{p_t:.4f}{'*' if p_t < 0.05 else ''}",
                f"{p_w:.4f}{'*' if p_w < 0.05 else ''}" if not np.isnan(p_w) else "N/A",
            ])

    headers = ["Comparison", "Metric", "Δ̄ (A−B)", "p-value (t-test)", "p-value (Wilcoxon)"]
    print(tabulate(rows, headers=headers, tablefmt="github"))
    print("\n* p < 0.05")


if __name__ == "__main__":
    main()