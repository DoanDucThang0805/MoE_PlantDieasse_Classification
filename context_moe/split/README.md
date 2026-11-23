# Frozen data split

Snapshot of the train/validation/test split produced by the existing code
(`src/utils/load_dataset.py`: stratified, `random_state=42`, 80/10/10).

- `plantdoc/` -> `data/tomato_only` (2274 / 284 / 285)
- `slif/` -> `data/slif_tomato_dataset/Phase_I-Dataset` (2136 / 267 / 267)
- Each folder: `train.csv`, `validation.csv`, `test.csv` (path relative to the data root, label, class_name)
  and `split_manifest.json` (counts, per-class counts, SHA-256 of each CSV).

```bash
python split/make_split.py [--dataset plantdoc|slif] [--force]   # (re)generate
python split/verify_split.py                                      # check CSVs + live code still agree
```

Needs scikit-learn (torch is optional). The training code still calls `LoadDataset`
directly; `verify_split.py` is what detects drift (different file list, sklearn version, ...).

## Pre-split image folders

```bash
python split/materialize_split.py [--dataset plantdoc|slif] [--link] [--force]
```

Builds `data/tomato_only_split/` and `data/slif_tomato_dataset/Phase_I-Dataset_split/`, each laid out as
`{train,validation,test}/<class>/<image>`, copied from the frozen CSVs (`--link` = symlinks).
`data/` is git-ignored, so these folders are not committed; re-run the script on a new machine.
