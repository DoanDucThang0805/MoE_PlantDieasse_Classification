"""Check that the frozen split still matches the dataset on disk and the live code.

Usage:
    python split/verify_split.py [--dataset plantdoc|slif|all]

Checks: CSV hashes, files exist, no overlap between splits, and that
LoadDataset (current code) still reproduces exactly the frozen lists.
Exit code 1 on any mismatch.
"""
import argparse
import json
import sys

from split_utils import DATASETS, SPLIT_DIR, SPLITS, current_split, read_csv, sha256


def verify(name: str) -> bool:
    d = SPLIT_DIR / name
    mf = d / "split_manifest.json"
    if not mf.exists():
        print(f"[MISSING] {name}: no frozen split (run make_split.py)")
        return False
    manifest = json.loads(mf.read_text(encoding="utf-8"))
    ok = True
    frozen = {}
    for s in SPLITS:
        f = d / manifest["splits"][s]["file"]
        if sha256(f) != manifest["splits"][s]["sha256"]:
            print(f"[FAIL] {name}/{s}: CSV hash changed"); ok = False
        frozen[s] = read_csv(f)
        missing = [p for p, _, _ in frozen[s] if not (DATASETS[name] / p).is_file()]
        if missing:
            print(f"[FAIL] {name}/{s}: {len(missing)} files missing, e.g. {missing[0]}"); ok = False
    sets = {s: {p for p, _, _ in frozen[s]} for s in SPLITS}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if sets[a] & sets[b]:
            print(f"[FAIL] {name}: overlap between {a} and {b}"); ok = False

    live, _ = current_split(name)
    for s in SPLITS:
        if live[s] != frozen[s]:
            print(f"[FAIL] {name}/{s}: live LoadDataset differs from frozen split "
                  f"(live n={len(live[s])}, frozen n={len(frozen[s])})"); ok = False
    if ok:
        print(f"[OK] {name}: " + ", ".join(f"{s}={len(frozen[s])}" for s in SPLITS))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    args = ap.parse_args()
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    sys.exit(0 if all([verify(n) for n in names]) else 1)


if __name__ == "__main__":
    main()
