"""Shared helpers for freezing / verifying the train/validation/test split.

The split itself is produced by the *existing* ``utils.load_dataset.LoadDataset``
(stratified 80/10/10, ``random_state=42``) -- nothing is re-implemented here.
"""
import csv
import hashlib
import sys
from pathlib import Path

SPLIT_DIR = Path(__file__).resolve().parent
ROOT = SPLIT_DIR.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch, torchvision  # noqa: F401
except ImportError:
    # The split logic only needs sklearn; stub torch so the real LoadDataset can still be
    # imported in an environment without torch (e.g. a bare sklearn env).
    import types
    _t = types.ModuleType("torch"); _t.Tensor = object; _t.tensor = None; _t.float32 = None
    _u = types.ModuleType("torch.utils"); _d = types.ModuleType("torch.utils.data"); _d.Dataset = object
    _tv = types.ModuleType("torchvision"); _tt = types.ModuleType("torchvision.transforms")
    _tt.transforms = types.SimpleNamespace(Compose=object); _tt.Compose = object
    sys.modules.update({"torch": _t, "torch.utils": _u, "torch.utils.data": _d,
                        "torchvision": _tv, "torchvision.transforms": _tt})
    _tt.transforms = _tt  # `from torchvision.transforms import transforms`

from utils.load_dataset import LoadDataset  # noqa: E402

SPLITS = ("train", "validation", "test")
TRAIN_RATIO = 0.8  # same value used in dataset/*_dataset.py

# dataset name -> data root (identical to the paths in src/dataset/*.py)
DATASETS = {
    "plantdoc": ROOT / "data" / "tomato_only",
    "slif": ROOT / "data" / "slif_tomato_dataset" / "Phase_I-Dataset",
}


def current_split(dataset: str):
    """Return {split: [(relative_path, label, class_name), ...]} from the live code."""
    root = DATASETS[dataset]
    out, class_to_idx = {}, None
    for split in SPLITS:
        ds = LoadDataset(root_dir=root, split=split, train_ratio=TRAIN_RATIO)
        idx_to_class = ds.idx_to_class
        class_to_idx = ds.class_to_idx
        out[split] = [
            (Path(p).relative_to(root).as_posix(), int(y), idx_to_class[int(y)])
            for p, y in zip(ds.image_paths, ds.labels)
        ]
    return out, class_to_idx


def write_csv(path: Path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "class_name"])
        w.writerows(rows)


def read_csv(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        return [(p, int(y), c) for p, y, c in r]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
