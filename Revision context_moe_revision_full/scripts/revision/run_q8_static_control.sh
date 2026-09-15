#!/usr/bin/env bash
set -euo pipefail
SEEDS=(42 43 44 45 46)
for seed in "${SEEDS[@]}"; do
  root="checkpoints/plantdoc/revision_controls/static_uniform_4expert/seed_${seed}"
  if [[ -f "$root/DONE" ]]; then
    echo "SKIP completed static control seed=$seed"
  else
    (cd src && python -m revision.train_static_uniform_control --dataset plantdoc --seed "$seed")
    mkdir -p "$root" && touch "$root/DONE"
  fi
done
# Evaluates all available revision controls; missing optional controls are reported, not fatal.
(cd src && python -m revision.evaluate_revision_controls)
