#!/bin/bash
# Inference script for MoE models
source venv/bin/activate

cd src

# ============================================================================
# Inference: 6 Experts (3 runs for each top_k)
# ============================================================================

echo "=========================================="
echo "Running inference for 6 Experts"
echo "=========================================="

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260408-175053 \
  --dataset-name plantdoc \
  --topk 6 \
  --num-expert 6

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260408-181936 \
  --dataset-name plantdoc \
  --topk 6 \
  --num-expert 6

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260408-183152 \
  --dataset-name plantdoc \
  --topk 6 \
  --num-expert 6

# ============================================================================
# Inference: 7 Experts (3 runs for each top_k from 1 to 7)
# ============================================================================

echo "=========================================="
echo "Running inference for 7 Experts"
echo "=========================================="

# Top K = 1
echo "--- 7 Experts, Top K = 1 ---"
python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260408-184824 \
  --dataset-name plantdoc \
  --topk 1 \
  --num-expert 7

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260408-191018 \
  --dataset-name plantdoc \
  --topk 1 \
  --num-expert 7

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260408-193052 \
  --dataset-name plantdoc \
  --topk 1 \
  --num-expert 7

# Top K = 2
echo "--- 7 Experts, Top K = 2 ---"
python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260408-194938 \
  --dataset-name plantdoc \
  --topk 2 \
  --num-expert 7

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260408-200650 \
  --dataset-name plantdoc \
  --topk 2 \
  --num-expert 7

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260408-204539 \
  --dataset-name plantdoc \
  --topk 2 \
  --num-expert 7

# Top K = 3
echo "--- 7 Experts, Top K = 3 ---"
python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260408-211657 \
  --dataset-name plantdoc \
  --topk 3 \
  --num-expert 7

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260408-213951 \
  --dataset-name plantdoc \
  --topk 3 \
  --num-expert 7

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260408-215635 \
  --dataset-name plantdoc \
  --topk 3 \
  --num-expert 7

# Top K = 4
echo "--- 7 Experts, Top K = 4 ---"
python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260408-221833 \
  --dataset-name plantdoc \
  --topk 4 \
  --num-expert 7

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260408-223444 \
  --dataset-name plantdoc \
  --topk 4 \
  --num-expert 7

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260408-225448 \
  --dataset-name plantdoc \
  --topk 4 \
  --num-expert 7

# Top K = 5
echo "--- 7 Experts, Top K = 5 ---"
python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260408-231114 \
  --dataset-name plantdoc \
  --topk 5 \
  --num-expert 7

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260409-002350 \
  --dataset-name plantdoc \
  --topk 5 \
  --num-expert 7

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260409-005833 \
  --dataset-name plantdoc \
  --topk 5 \
  --num-expert 7

# Top K = 6
echo "--- 7 Experts, Top K = 6 ---"
python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260409-012838 \
  --dataset-name plantdoc \
  --topk 6 \
  --num-expert 7

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260409-013828 \
  --dataset-name plantdoc \
  --topk 6 \
  --num-expert 7

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260409-015542 \
  --dataset-name plantdoc \
  --topk 6 \
  --num-expert 7

# Top K = 7
echo "--- 7 Experts, Top K = 7 ---"
python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260409-020923 \
  --dataset-name plantdoc \
  --topk 7 \
  --num-expert 7

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260409-022207 \
  --dataset-name plantdoc \
  --topk 7 \
  --num-expert 7

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260409-023842 \
  --dataset-name plantdoc \
  --topk 7 \
  --num-expert 7

# ============================================================================
# Inference: 8 Experts (3 runs for each top_k from 1 to 8)
# ============================================================================

echo "=========================================="
echo "Running inference for 8 Experts"
echo "=========================================="

# Top K = 1
echo "--- 8 Experts, Top K = 1 ---"
python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260409-025609 \
  --dataset-name plantdoc \
  --topk 1 \
  --num-expert 8

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260409-030847 \
  --dataset-name plantdoc \
  --topk 1 \
  --num-expert 8

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260409-031805 \
  --dataset-name plantdoc \
  --topk 1 \
  --num-expert 8

# Top K = 2
echo "--- 8 Experts, Top K = 2 ---"
python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260409-033623 \
  --dataset-name plantdoc \
  --topk 2 \
  --num-expert 8

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260409-034943 \
  --dataset-name plantdoc \
  --topk 2 \
  --num-expert 8

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260409-040144 \
  --dataset-name plantdoc \
  --topk 2 \
  --num-expert 8

# Top K = 3
echo "--- 8 Experts, Top K = 3 ---"
python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260409-041948 \
  --dataset-name plantdoc \
  --topk 3 \
  --num-expert 8

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260409-043309 \
  --dataset-name plantdoc \
  --topk 3 \
  --num-expert 8

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260409-045358 \
  --dataset-name plantdoc \
  --topk 3 \
  --num-expert 8

# Top K = 4
echo "--- 8 Experts, Top K = 4 ---"
python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260409-050836 \
  --dataset-name plantdoc \
  --topk 4 \
  --num-expert 8

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260409-052524 \
  --dataset-name plantdoc \
  --topk 4 \
  --num-expert 8

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260409-054033 \
  --dataset-name plantdoc \
  --topk 4 \
  --num-expert 8

# Top K = 5
echo "--- 8 Experts, Top K = 5 ---"
python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260409-055501 \
  --dataset-name plantdoc \
  --topk 5 \
  --num-expert 8

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260409-061453 \
  --dataset-name plantdoc \
  --topk 5 \
  --num-expert 8

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260409-063017 \
  --dataset-name plantdoc \
  --topk 5 \
  --num-expert 8

# Top K = 6
echo "--- 8 Experts, Top K = 6 ---"
python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260409-064436 \
  --dataset-name plantdoc \
  --topk 6 \
  --num-expert 8

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260409-070152 \
  --dataset-name plantdoc \
  --topk 6 \
  --num-expert 8

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260409-072054 \
  --dataset-name plantdoc \
  --topk 6 \
  --num-expert 8

# Top K = 7
echo "--- 8 Experts, Top K = 7 ---"
python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260409-072959 \
  --dataset-name plantdoc \
  --topk 7 \
  --num-expert 8

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260409-073918 \
  --dataset-name plantdoc \
  --topk 7 \
  --num-expert 8

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260409-075706 \
  --dataset-name plantdoc \
  --topk 7 \
  --num-expert 8

# Top K = 8
echo "--- 8 Experts, Top K = 8 ---"
python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260409-081035 \
  --dataset-name plantdoc \
  --topk 8 \
  --num-expert 8

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260409-082354 \
  --dataset-name plantdoc \
  --topk 8 \
  --num-expert 8

python -m inference.moe.inference \
  --model-name mobilenetv3small_moe \
  --run-time run_20260409-083356 \
  --dataset-name plantdoc \
  --topk 8 \
  --num-expert 8

echo "=========================================="
echo "All inference completed!"
echo "=========================================="