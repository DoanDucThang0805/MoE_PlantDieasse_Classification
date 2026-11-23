"""Freeze the current train/validation/test split to CSV files.

Usage (from anywhere):
    python split/make_split.py                  # both datasets
    python split/make_split.py --dataset plantdoc
    python split/make_split.py --force          # overwrite an existing frozen split

Output: split/<dataset>/{train,validation,test}.csv + split_manifest.json
Paths in the CSVs are relative to the dataset root, so they are machine-independent.
"""
import argparse
import json
from collections import Counter

import sklearn

from split_utils import DATASETS, SPLIT_DIR, SPLITS, TRAIN_RATIO, current_split, sha256, write_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    ap.add_argument("--force", action="store_true", help="overwrite an existing frozen split")
    args = ap.parse_args()

    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for name in names:
        out_dir = SPLIT_DIR / name
        manifest_path = out_dir / "split_manifest.json"
        if manifest_path.exists() and not args.force:
            print(f"[SKIP] {name}: already frozen at {out_dir} (use --force to overwrite)")
            continue
        if not DATASETS[name].is_dir():
            print(f"[SKIP] {name}: data root not found: {DATASETS[name]}")
            continue

        splits, class_to_idx = current_split(name)
        out_dir.mkdir(exist_ok=True)
        manifest = {
            "dataset": name,
            "data_root": DATASETS[name].relative_to(SPLIT_DIR.parent).as_posix(),
            "method": "sklearn.train_test_split, stratified, shuffle=True, random_state=42; "
                      "train 80% / validation 10% / test 10% (utils.load_dataset.LoadDataset)",
            "train_ratio": TRAIN_RATIO,
            "sklearn_version_at_freeze": sklearn.__version__,
            "class_to_idx": class_to_idx,
            "splits": {},
        }
        for s in SPLITS:
            f = out_dir / f"{s}.csv"
            write_csv(f, splits[s])
            manifest["splits"][s] = {
                "file": f.name,
                "n": len(splits[s]),
                "per_class": dict(sorted(Counter(r[2] for r in splits[s]).items())),
                "sha256": sha256(f),
            }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] {name}: " + ", ".join(f"{s}={manifest['splits'][s]['n']}" for s in SPLITS))


if __name__ == "__main__":
    main()
