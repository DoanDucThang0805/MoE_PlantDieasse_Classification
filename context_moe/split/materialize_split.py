"""Build a physical pre-split dataset from the frozen CSVs.

    python split/materialize_split.py [--dataset plantdoc|slif|all] [--link] [--force]

Creates  data/<root_name>_split/{train,validation,test}/<class_name>/<image>
(plantdoc -> data/tomato_only_split, slif -> data/slif_tomato_dataset/Phase_I-Dataset_split)
Default is copy; --link uses symlinks (saves disk, breaks if the source moves).
Files are read from the frozen CSVs, never re-split.
"""
import argparse
import shutil
import sys

from split_utils import DATASETS, SPLIT_DIR, SPLITS, read_csv


def build(name: str, link: bool, force: bool) -> bool:
    src_root = DATASETS[name]
    out_root = src_root.parent / (src_root.name + "_split")
    if out_root.exists():
        if not force:
            print(f"[SKIP] {name}: {out_root} exists (use --force to rebuild)")
            return True
        shutil.rmtree(out_root)
    total = 0
    for s in SPLITS:
        rows = read_csv(SPLIT_DIR / name / f"{s}.csv")
        for rel, _, cls in rows:
            src = src_root / rel
            dst = out_root / s / cls / src.name
            if dst.exists():
                print(f"[FAIL] {name}/{s}: duplicate filename {cls}/{src.name}")
                return False
            dst.parent.mkdir(parents=True, exist_ok=True)
            if link:
                dst.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dst)
        total += len(rows)
        print(f"  {s}: {len(rows)}")
    print(f"[OK] {name}: {total} images -> {out_root}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    ap.add_argument("--link", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    names = list(DATASETS) if a.dataset == "all" else [a.dataset]
    sys.exit(0 if all([build(n, a.link, a.force) for n in names]) else 1)


if __name__ == "__main__":
    main()
