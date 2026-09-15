#!/usr/bin/env bash
set -euo pipefail
# Run from repository root. A DONE marker is written only after a training process
# exits normally, so an interrupted run will be retrained rather than silently skipped.
SEEDS=(42 43 44 45 46)
for k in 1 2 3 4; do
  for seed in "${SEEDS[@]}"; do
    root="checkpoints/plantdoc/revision_topk_linear/mobilenetv3small_moe/4_experts/top_${k}/seed_${seed}"
    if [[ -f "$root/DONE" ]]; then
      echo "SKIP completed k=$k seed=$seed"
    else
      (cd src && python -m revision.train_moe_controlled --dataset plantdoc --num_experts 4 --top_k "$k" --seed "$seed")
      mkdir -p "$root" && touch "$root/DONE"
    fi
  done
done
(cd src && python -m revision.evaluate_topk --dataset plantdoc)
