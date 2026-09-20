# PlantDoc revision evidence package

This directory is a frozen snapshot of the numerical evidence used in
`baocaorevisionmoe.tex` on 2026-09-20. Reported classification results are
the mean and sample standard deviation over seeds 42--46 and are rounded to
four decimal places in the report. Raw CSV precision is intentionally
preserved in this package.

The package excludes paired-test and per-seed discussion that is not used in
the current report. It also excludes the older PDF because it predates the
latest TeX edits; `report/baocaorevisionmoe.tex` is the authoritative report
snapshot in this directory.

## Evidence map

### Run 1: model and complexity audit

Directory: `run_01_model_audit/`

- `model_audit.csv`: parameter breakdown from code and checkpoint state.
- `active_complexity_thop.csv`: THOP active parameters and FLOPs for
  Top-k = 1, 2, 3, 4.
- Reported Top-2 values: 5.8535 M stored parameters, 3.4845 M activated
  parameters, and 0.0634 GFLOPs.

The representative audit checkpoint is:

`context_moe/checkpoints/plantdoc/moe_linearcontextaware_temp0.5/mobilenetv3small_moe/4_experts/top_2/seed_42/run_20260504-171527/best_checkpoint.pth`

SHA-256: `756c0094a34436f16cae4c48d5c8d0d852fe64af85b0882d83e72d562601bc85`

Its metadata is: 8 classes, 4 experts, Top-2, temperature 0.5,
context-aware routing, and a 6-dimensional context vector.

### Run 2: routing-utilization audit

Directory: `run_02_routing_usage/`

- `global_selection_share_per_seed.csv`: raw values for five seeds.
- `global_selection_share_summary.csv`: mean and standard deviation used in
  the report.
- `class_activation_rate_per_seed.csv`: raw class-wise activation rates.
- `class_activation_rate_summary.csv`: mean and standard deviation used in
  the report.

The global expert shares are 0.2446, 0.2554, 0.2495, and 0.2505. Class
activation rates are selection frequencies, not routing weights.

### Run 3: shuffled-routing counterfactual

Directory: `run_03_shuffled_routing/`

- `shuffle_permutations.csv`: 100 permutations for each of five checkpoints.
- `shuffle_seed_summary.csv`: checkpoint-level summaries used to calculate
  the five-seed mean and standard deviation.

Reported results:

| Metric | Learned routing | Shuffled routing | Mean difference |
|---|---:|---:|---:|
| Accuracy | 0.8596 +/- 0.0124 | 0.8287 +/- 0.0102 | +0.0309 |
| Macro-F1 | 0.8275 +/- 0.0153 | 0.7912 +/- 0.0129 | +0.0363 |

### Run 4: PyTorch eager runtime on Raspberry Pi 5

Directory: `run_04_pytorch_runtime_pi/`

- `sparse_runtime_pi.csv`: mean, median, P95, and standard deviation from 300
  timed forwards after 30 warm-up forwards.
- `system_info.json`: device, software, benchmark settings, and checkpoint
  hash.
- `raspberry_pi_status.txt`: temperature and throttling state.
- `pi_benchmark_results.zip`: original bundle returned by the Raspberry Pi.
- `sparse_runtime_pi_derived.csv`: supplementary derived columns not used in
  the current report table.

This benchmark executed the PyTorch eager model directly on the Raspberry Pi
5 CPU. It is not an ONNX Runtime measurement. The runtime checkpoint is:

`context_moe/checkpoints/plantdoc/moe_linearcontextaware_temp0.5/mobilenetv3small_moe/4_experts/top_2/seed_42/run_20260920-110525/best_checkpoint.pth`

SHA-256: `e7683d54bc2457c8ae4664db14fcf271a2a940b71d01b66f8895fc0fc5cc3678`

### Run 5: controlled Top-k ablation

Directory: `run_05_controlled_topk/`

- `topk_seed_metrics.csv`: Accuracy, Macro-F1, and checkpoint provenance for
  every k and seed.
- `topk_summary.csv`: five-seed mean and standard deviation reported in the
  paper.
- `topk_training_k134_20260915.log`: training completion record for the newly
  trained k = 1, 3, and 4 controls.

| Top-k | Accuracy | Macro-F1 |
|---:|---:|---:|
| 1 | 0.8309 +/- 0.0169 | 0.7927 +/- 0.0126 |
| 2 | 0.8596 +/- 0.0124 | 0.8275 +/- 0.0153 |
| 3 | 0.8540 +/- 0.0231 | 0.8243 +/- 0.0268 |
| 4 | 0.8491 +/- 0.0197 | 0.8124 +/- 0.0291 |

### Run 6: static-uniform four-expert control

Directory: `run_06_static_uniform/`

- `control_seed_metrics.csv`: raw five-seed metrics.
- `control_summary.csv`: five-seed mean and standard deviation.
- `control_complexity_thop.csv`: THOP complexity for the compared models.

Main MoE reaches 0.8596 +/- 0.0124 Accuracy and 0.8275 +/- 0.0153 Macro-F1.
Static-uniform reaches 0.8365 +/- 0.0234 Accuracy and
0.8019 +/- 0.0247 Macro-F1.

### Run 7: six-dimensional context ablation

Directory: `run_07_context_ablation/`

- `context_seed_metrics.csv`: raw metrics for the main context-aware model
  and the independently trained zero-context control over seeds 42--46.

Main context-aware MoE reaches 0.8596 +/- 0.0124 Accuracy and
0.8275 +/- 0.0153 Macro-F1. Zero-context reaches 0.8330 +/- 0.0343 Accuracy
and 0.8018 +/- 0.0342 Macro-F1.

### Run 8: matched torchvision MobileNetV3-Small baseline

Directory: `run_08_matched_baseline/`

- `matched_mobilenetv3_checkpoint_provenance.csv`: exact checkpoint used for
  each seed.
- `control_seed_metrics.csv`: raw five-seed baseline metrics.
- `control_summary.csv`: mean and standard deviation.
- `control_complexity_thop.csv`: THOP model complexity.

The matched baseline reaches 0.8267 +/- 0.0223 Accuracy and
0.7894 +/- 0.0242 Macro-F1.

## Dataset and reporting convention

The PlantDoc tomato subset contains 2,843 images from eight classes. The
fixed split contains 2,274 training, 284 validation, and 285 test images.
Accuracy and Macro-F1 are reported as mean +/- sample standard deviation over
five seeds. Run 1 reports deterministic architecture/complexity values. Run
4 reports timing statistics over 300 repetitions and therefore is not a
five-seed classification result.
