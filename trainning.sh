#!/bin/bash

# load conda
source venv/bin/activate

cd src

# ============================================================================
# Script Huấn Luyện Mô Hình MoE với Nhiều Cấu Hình
# ============================================================================
# Cấu hình: num_experts chạy từ 2 đến 8
#          top_k chạy từ 1 đến num_experts
#          Mỗi cấu hình chạy 3 lần

for num_experts in {5..8}; do
    for top_k in $(seq 1 $num_experts); do
        for run in {1..3}; do
            clear
            echo "=========================================="
            echo "Training: num_experts=$num_experts, top_k=$top_k"
            echo "Run $run/3"
            echo "=========================================="
            
            PYTHONPATH=src python -m trainning.moe_train\
                --num_experts $num_experts\
                --top_k $top_k\
                --num_epoch 300
            
            # Kiểm tra xem quá trình training có bị lỗi không
            if [ $? -ne 0 ]; then
                echo "ERROR: Training failed for num_experts=$num_experts, top_k=$top_k, run=$run"
                exit 1
            fi
        done
    done
done

echo "=========================================="
echo "All training completed successfully!"
echo "=========================================="