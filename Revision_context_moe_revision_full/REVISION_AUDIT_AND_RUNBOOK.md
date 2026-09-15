# Revision audit and minimum experiment package

## Confirmed from the supplied code/results

1. **Current proposed Linear-Gating MoE is not 3.4845 M parameters.** The code uses four `576 -> 1024 -> 576` experts. Structural count is approximately **5,853,496 parameters** for `E=4`, plus/minus only if the checked-in backbone implementation changes. Run `revision.model_consistency_audit` on an actual checkpoint to record the exact state.
2. **PyTorch MoE forward is dynamically sparse.** `linear_model.py` only evaluates an expert for samples selected by Top-k.
3. **Current ONNX export is deliberately dense.** `model_adapt_onnx.py` stacks `[expert(x) for expert in self.experts]` before masking. Therefore the reported Raspberry Pi ONNX latency is a **dense-export deployment measurement**, not evidence that only k experts are executed on the Pi.
4. **The load-balance loss is already implemented and can be stated exactly:** `L_bal = E * sum_e f_e P_e`, where `P_e` is the batch mean softmax router probability and `f_e` is the normalized selection share (`selections/(B*k)`). With balanced routing its minimum is 1.
5. **Q7 normalization is mathematically consistent in code but must be named correctly.** Global usage divides by `N*k`, hence sums to 1. Class-wise activation divides by `N_c`, hence each class row sums to `k`. The class heatmap is an activation rate, not a routing weight.
6. **The old Top-k table does not answer Q4 for the main Linear Gate.** `results/moe_contextaware_temp0.5.csv` belongs to the MLP context-aware gate. The main Linear Gate file contains only `E=4,k=2`.
7. **The older 3.16 M / 3.33 M controls are not parameter-matched to the current 5.85 M MoE.** Keep them as architectural reference controls, but do not call them parameter-matched after correcting the MoE parameter count.
8. **Statistical code intentionally leaves the four primary tests unadjusted and applies Holm/BH to the 20 exploratory tests only.** The manuscript must say this explicitly; do not say Holm correction was applied to all 24 tests.
9. **Legacy training/evaluation scripts are dataset-ambiguous.** The checked-in `trainning/moe_train.py` imports SLIF directly, while some result files are PlantDoc. Revision scripts added here require an explicit `--dataset` argument.

## Minimum new work

### Q4: controlled Top-k ablation
Train the main **Linear context-aware gate** with fixed `E=4`, `k in {1,2,3,4}`, seeds 42--46. Use `scripts/revision/run_q4_topk.sh`.

### Q8: two complementary controls
A. Train `StaticUniformExpertModel`: same backbone, same four `576->1024->576` experts, same residual and final classifier, but no input-dependent gate. The parameter difference from MoE is only the small router (~8.6k, about 0.15%). Use `scripts/revision/run_q8_static_control.sh`.

B. Run `revision.routing_counterfactual` on the existing five main checkpoints. It permutes learned complete Top-k route decisions across test images while preserving the empirical route distribution. This tests whether input-route correspondence contributes beyond stored capacity. It does **not** prove semantic specialization.

### Q1: execution audit
Run `revision.sparse_runtime_benchmark` on Raspberry Pi using PyTorch eager/CPU. Keep the existing ONNX numbers, but label them as dense-export deployment results unless a genuinely sparse backend/export is implemented.

### Q7: corrected terminology and multi-seed reporting
Run `revision.routing_utilization_audit`; report global selection share and class activation rate separately. Do not call the heatmap values routing weights.

## Suggested commands

From repository root:

```bash
# Exact checkpoint architecture audit
cd src
python -m revision.model_consistency_audit \
  --checkpoint ../checkpoints/plantdoc/moe_linearcontextaware_temp0.5/mobilenetv3small_moe/4_experts/top_2/seed_42/<run>/best_checkpoint.pth \
  --output_csv ../diagnostics/revision/model_audit.csv

# Q7 across five seeds
python -m revision.routing_utilization_audit \
  --dataset plantdoc \
  --checkpoint_root ../checkpoints/plantdoc/moe_linearcontextaware_temp0.5/mobilenetv3small_moe/4_experts/top_2

# Q8: no retraining, shuffled-route counterfactual
python -m revision.routing_counterfactual \
  --dataset plantdoc \
  --checkpoint_root ../checkpoints/plantdoc/moe_linearcontextaware_temp0.5/mobilenetv3small_moe/4_experts/top_2 \
  --permutations 100

# Q1 true sparse PyTorch runtime on Pi
python -m revision.sparse_runtime_benchmark \
  --checkpoint ../checkpoints/plantdoc/moe_linearcontextaware_temp0.5/mobilenetv3small_moe/4_experts/top_2/seed_42/<run>/best_checkpoint.pth \
  --threads 1 --runs 300
```

## Manuscript claim boundary after these tests

Do not claim a novel router. Do not claim lower overall cost than MobileNetV3-Small. Do not infer semantic specialization from activation heatmaps alone. The defensible contribution is an input-dependent sparse classifier-side capacity mechanism whose accuracy/robustness benefit and parameter/runtime trade-off are explicitly measured.

## Additional implementation/manuscript mismatch: the 6-D context path

The main training code instantiates `ContextAwareLinearGating`. Its gate receives the MobileNet feature **and** six image-derived context variables from `utils/context_features.py`: mean brightness, brightness standard deviation, blur score, edge density, mean saturation, and green ratio. The supplied manuscript PDF instead writes `g_phi(u_i)` and does not document this second input path.

This is not a cosmetic issue. The manuscript must either (i) disclose the 6-D context vector, add it to the architecture/equations, and correct the gate parameter count, or (ii) retrain a feature-only router. As a minimum sensitivity check, `train_zero_context_control.py` keeps the same architecture and parameter count but zeros all six context variables. Run `scripts/revision/run_zero_context.sh` for seeds 42--46.

For the implemented Linear context-aware gate with `d=576`, `c=6`, projected context width `r_c=32`, `E=4`, the gate contains **8,592 parameters**, not `Ed+E=2,308`. The manuscript complexity equation for the gate therefore needs revision if the context-aware implementation is retained.
