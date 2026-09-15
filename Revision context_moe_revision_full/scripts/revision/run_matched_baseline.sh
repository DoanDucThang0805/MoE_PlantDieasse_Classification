#!/usr/bin/env bash
set -euo pipefail
for seed in 42 43 44 45 46; do
  root="checkpoints/plantdoc/revision_controls/matched_torchvision_mnv3small/seed_${seed}"
  if [[ -f "$root/DONE" ]]; then
    echo "SKIP completed matched baseline seed=$seed"
  else
    (cd src && python -m revision.train_matched_backbone_baseline --dataset plantdoc --seed "$seed")
    mkdir -p "$root" && touch "$root/DONE"
  fi
done
(cd src && python -m revision.evaluate_revision_controls)
