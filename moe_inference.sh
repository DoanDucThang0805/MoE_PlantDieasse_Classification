#!/bin/bash
# Batch inference script for all MoE models
# Automatically discovers and runs inference on all model variants in:
# checkpoints/plantdoc/moe_contextaware_temp1.0/mobilenetv3small_moe

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
fi

cd src
clear

echo "=========================================="
echo "Batch Inference - All MoE Models"
echo "=========================================="
echo ""

CHECKPOINT_ROOT="../checkpoints/plantdoc/moe_contextaware_temp0.5/mobilenetv3small_moe"
DATASET_NAME="plantdoc"
MODEL_NAME="mobilenetv3small_moe"
TYPE_MODEL="moe_contextaware_temp0.5"

# Check if checkpoint directory exists
if [ ! -d "$CHECKPOINT_ROOT" ]; then
    echo "ERROR: Checkpoint directory not found: $CHECKPOINT_ROOT"
    echo "Current directory: $(pwd)"
    echo "Available directories in ..: $(ls -la ../ 2>/dev/null | grep drwx)"
    exit 1
fi

# Counter
total=0
success=0
failed=0

# Arrays to store results
declare -a failed_models

echo "Discovering models in: $CHECKPOINT_ROOT"
echo "Full path: $(cd "$CHECKPOINT_ROOT" 2>/dev/null && pwd)"
echo ""

# Iterate through all experts (2_experts, 3_experts, ..., 8_experts)
for experts_dir in "$CHECKPOINT_ROOT"/*/; do
    experts_name=$(basename "$experts_dir")
    # Remove trailing slash and check pattern
    experts_name="${experts_name%/}"
    
    if ! [[ $experts_name =~ ^[0-9]+_experts$ ]]; then
        continue
    fi
    
    num_experts=${experts_name%_*}
    
    # Iterate through top_k (top_1, top_2, ...)
    for top_k_dir in "$experts_dir"*/; do
        [ ! -d "$top_k_dir" ] && continue
        top_k_name=$(basename "$top_k_dir")
        top_k_name="${top_k_name%/}"
        
        if ! [[ $top_k_name =~ ^top_[0-9]+$ ]]; then
            continue
        fi
        
        top_k=${top_k_name#top_}
        
        # Iterate through seeds (seed_42, seed_43, ...)
        for seed_dir in "$top_k_dir"*/; do
            [ ! -d "$seed_dir" ] && continue
            seed_name=$(basename "$seed_dir")
            seed_name="${seed_name%/}"
            
            if ! [[ $seed_name =~ ^seed_[0-9]+$ ]]; then
                continue
            fi
            
            seed=${seed_name#seed_}
            
            # Iterate through run_time folders (run_20260427-223951, ...)
            for run_time_dir in "$seed_dir"*/; do
                [ ! -d "$run_time_dir" ] && continue
                run_time=$(basename "$run_time_dir")
                run_time="${run_time%/}"
                
                if ! [[ $run_time =~ ^run_[0-9]+-[0-9]+$ ]]; then
                    continue
                fi
                
                checkpoint_file="$run_time_dir/best_checkpoint.pth"
                if [ ! -f "$checkpoint_file" ]; then
                    continue
                fi
                
                ((total++))
                
                echo "[Model $total] Running: experts=$num_experts, top_k=$top_k, seed=$seed, run_time=$run_time"
                
                # Run inference (show output for debugging)
                if python -m inference.moe.context_aware_moe_inference \
                    --model_name "$MODEL_NAME" \
                    --type_model "$TYPE_MODEL" \
                    --dataset_name "$DATASET_NAME" \
                    --num_experts "$num_experts" \
                    --top_k "$top_k" \
                    --seed "$seed" \
                    --run_time "$run_time" \
                    --use_context \
                    --router_mode context_aware; then
                    
                    ((success++))
                    echo "  ✓ Success"
                else
                    ((failed++))
                    failed_models+=("experts=$num_experts, top_k=$top_k, seed=$seed, run_time=$run_time")
                    echo "  ✗ Failed"
                fi
                
                echo ""
            done
        done
    done
done

# Print summary
echo "=========================================="
echo "BATCH INFERENCE SUMMARY"
echo "=========================================="
echo "Total models processed: $total"

if [ $total -eq 0 ]; then
    echo ""
    echo "WARNING: No models found! Checking directory structure..."
    echo "Checkpoint root: $CHECKPOINT_ROOT"
    
    if [ -d "$CHECKPOINT_ROOT" ]; then
        echo "Directory exists. Contents:"
        ls -la "$CHECKPOINT_ROOT" | head -20
    else
        echo "Directory does not exist!"
    fi
    
    exit 1
fi

echo "Successful: $success/$total"
echo "Failed: $failed/$total"
echo ""

if [ $failed -gt 0 ]; then
    echo "Failed models:"
    for model in "${failed_models[@]}"; do
        echo "  - $model"
    done
fi

echo "=========================================="
echo "Batch inference completed!"
echo "=========================================="

exit $failed
