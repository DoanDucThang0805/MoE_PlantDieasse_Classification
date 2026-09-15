#!/usr/bin/env bash
set -euo pipefail
for seed in 42 43 44 45 46; do
  root="checkpoints/plantdoc/revision_controls/moe_zero_context/4_experts/top_2/seed_${seed}"
  if [[ -f "$root/DONE" ]]; then
    echo "SKIP completed zero-context seed=$seed"
  else
    (cd src && python -m revision.train_zero_context_control --dataset plantdoc --seed "$seed")
    mkdir -p "$root" && touch "$root/DONE"
  fi
done
(cd src && python -m revision.evaluate_zero_context)
