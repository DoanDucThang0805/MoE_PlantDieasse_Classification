# HƯỚNG DẪN CHẠY BỔ SUNG CHO REVISION MoE–MobileNetV3

Tài liệu này áp dụng cho gói `context_moe_revision_full.zip` đã được audit. Mục tiêu là tạo đúng các bằng chứng cần thiết cho Q1, Q4, Q7 và Q8 mà không mở rộng thí nghiệm không cần thiết. Cấu hình mặc định bên dưới dùng **PlantDoc** vì các checkpoint chính trong code được tổ chức theo nhánh này. Nếu cần lặp lại trên SLIF, xem Mục 11.

## 1. Các mục tiêu và thứ tự ưu tiên

Chạy theo thứ tự sau:

1. **Audit checkpoint/parameter count** — không train.
2. **Q7 routing-utilization audit** — không train.
3. **Q8 shuffled-routing counterfactual** — không train.
4. **Q1 dynamic sparse runtime** — không train; nên chạy thêm trên Raspberry Pi 5.
5. **Q4 controlled Top-k** — train `E=4, k=1,2,3,4`, 5 seeds.
6. **Q8 static-uniform 4-expert control** — train 5 seeds.
7. **6-D context ablation** — train 5 seeds; cần thiết để giải quyết khác biệt giữa implementation và manuscript.
8. **Matched torchvision MobileNetV3-Small** — tùy chọn nhưng nên chạy nếu muốn có baseline đồng nhất implementation.

Không cần chạy lại sweep `E=5..8`, không thêm dataset mới và không thêm backbone mới cho mục tiêu phản biện hiện tại.

---

## 2. Chuẩn bị thư mục

Giải nén gói và đứng tại thư mục gốc repository. Cấu trúc tối thiểu cần có:

```text
repository/
├── data/
│   ├── tomato_only/                         # PlantDoc tomato subset
│   └── slif_tomato_dataset/Phase_I-Dataset/ # nếu chạy SLIF
├── checkpoints/
├── diagnostics/
├── results/
├── scripts/revision/
└── src/
    ├── revision/
    ├── models/
    ├── dataset/
    └── utils/
```

Dữ liệu PlantDoc được code đọc trực tiếp tại `data/tomato_only/`. Không đổi đường dẫn trong script nếu thư mục đã có đúng cấu trúc này.

Các checkpoint MoE chính đã huấn luyện trước đó cần được đặt theo dạng:

```text
checkpoints/plantdoc/
└── moe_linearcontextaware_temp0.5/
    └── mobilenetv3small_moe/
        └── 4_experts/
            └── top_2/
                ├── seed_42/run_*/best_checkpoint.pth
                ├── seed_43/run_*/best_checkpoint.pth
                ├── seed_44/run_*/best_checkpoint.pth
                ├── seed_45/run_*/best_checkpoint.pth
                └── seed_46/run_*/best_checkpoint.pth
```

Các bước 1–4 cần checkpoint này. Nếu chưa copy checkpoint vào repository, các script sẽ báo `[MISSING]` hoặc `FileNotFoundError`; không nên thay bằng checkpoint từ MLP gate.

---

## 3. Môi trường Python

Khuyến nghị tạo môi trường riêng. Ví dụ Linux/WSL:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Cài PyTorch phù hợp với GPU/CUDA của máy. Nếu máy đang dùng CUDA 12.8 và driver tương thích:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements_revision.txt
```

Nếu dùng CPU hoặc CUDA khác, cài `torch`/`torchvision` theo đúng build tương ứng rồi mới chạy dòng thứ hai. Không nên ép `cu128` nếu driver không phù hợp.

Kiểm tra nhanh:

```bash
python -c "import torch, torchvision, cv2, timm, scipy; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Sau đó kiểm tra dataset từ thư mục gốc:

```bash
cd src
python -m dataset.plantdoc_dataset
cd ..
```

Phải nhìn thấy số lượng train/validation/test và mapping các class; không được có `FileNotFoundError` ở `data/tomato_only`.

---

## 4. Bước A — audit architecture và parameter count [BẮT BUỘC, KHÔNG TRAIN]

Tìm một checkpoint chính, ví dụ seed 42:

```bash
find checkpoints/plantdoc/moe_linearcontextaware_temp0.5/mobilenetv3small_moe/4_experts/top_2/seed_42 \
  -name best_checkpoint.pth -print
```

Sau đó, từ `src/`:

```bash
cd src
python -m revision.model_consistency_audit \
  --checkpoint ../checkpoints/plantdoc/moe_linearcontextaware_temp0.5/mobilenetv3small_moe/4_experts/top_2/seed_42/run_XXXX/best_checkpoint.pth \
  --output_csv ../diagnostics/revision/model_audit.csv
cd ..
```

Thay `run_XXXX` bằng thư mục thật.

### File cần giữ

```text
diagnostics/revision/model_audit.csv
```

### Kiểm tra pass/fail

- `experts_total` phải tương ứng bốn expert `576 -> 1024 -> 576`.
- `linear_context_gate` phải phản ánh gate có context 6-D.
- `state_dict_tensor_total` phải phù hợp về cấu trúc với model hiện tại.
- Nếu checkpoint không load hoặc metadata khác `num_experts=4`, `top_k=2`, `context_dim=6`, dừng trước khi chạy các bước sau và xác định đúng checkpoint.

Lưu ý: `state_dict_tensor_total` có thể chứa buffer, vì vậy khi đưa số **trainable parameters** vào manuscript phải ưu tiên `sum(p.numel() for p in model.parameters())` của model instantiated, không dùng mù quáng tổng tensor state dict.

---

## 5. Bước B — Q7: expert utilization [BẮT BUỘC, KHÔNG TRAIN]

Từ `src/`:

```bash
cd src
python -m revision.routing_utilization_audit \
  --dataset plantdoc \
  --checkpoint_root ../checkpoints/plantdoc/moe_linearcontextaware_temp0.5/mobilenetv3small_moe/4_experts/top_2 \
  --output_dir ../diagnostics/revision/routing_usage
cd ..
```

### File đầu ra

```text
diagnostics/revision/routing_usage/
├── global_selection_share_per_seed.csv
├── global_selection_share_summary.csv
├── class_activation_rate_per_seed.csv
└── class_activation_rate_summary.csv
```

### Kiểm tra bắt buộc

Console phải cho, trong sai số số học:

```text
global sum ≈ 1.0
class row sums ≈ 2.0
```

vì cấu hình chính là Top-2.

Trong manuscript dùng đúng thuật ngữ:

- `global_selection_share`: tổng qua expert bằng 1;
- `class_activation_rate`: tổng một hàng class bằng `k=2`;
- không gọi `class_activation_rate` là “routing weight”.

Nếu một expert gần 0 ở **global selection share** trên nhiều seeds, cần xem lại load balancing. Việc một expert thấp đối với riêng một class không tự động là expert collapse.

---

## 6. Bước C — Q8: shuffled-routing counterfactual [BẮT BUỘC, KHÔNG TRAIN]

Đây là thí nghiệm chi phí thấp nhất để kiểm tra liệu ánh xạ **input → route** có thực sự đóng góp hay không.

```bash
cd src
python -m revision.routing_counterfactual \
  --dataset plantdoc \
  --checkpoint_root ../checkpoints/plantdoc/moe_linearcontextaware_temp0.5/mobilenetv3small_moe/4_experts/top_2 \
  --permutations 100 \
  --output_dir ../diagnostics/revision/routing_counterfactual
cd ..
```

### File đầu ra

```text
diagnostics/revision/routing_counterfactual/
├── shuffle_permutations.csv
├── shuffle_seed_summary.csv
└── shuffle_paired_statistics.csv
```

### Cách đọc

Quan tâm hai đại lượng:

```text
delta_accuracy = learned_accuracy - shuffled_accuracy_mean
delta_macro_f1 = learned_macro_f1 - shuffled_macro_f1_mean
```

Nếu delta dương nhất quán qua 5 seeds, có thể kết luận **input-dependent routing contributes to performance**. Không suy diễn thành “semantic specialization”.

Với chỉ 5 seeds, báo cả effect/delta và từng seed; không chỉ dựa vào `p < 0.05`. Wilcoxon với `n=5` có công suất thống kê rất hạn chế.

---

## 7. Bước D — Q1: dynamic sparse runtime [BẮT BUỘC, KHÔNG TRAIN]

Chạy trước trên máy phát triển để kiểm tra script; sau đó chạy cùng lệnh trên Raspberry Pi 5 để có số deployment phù hợp.

```bash
cd src
python -m revision.sparse_runtime_benchmark \
  --checkpoint ../checkpoints/plantdoc/moe_linearcontextaware_temp0.5/mobilenetv3small_moe/4_experts/top_2/seed_42/run_XXXX/best_checkpoint.pth \
  --threads 1 \
  --warmup 30 \
  --runs 300 \
  --output_csv ../diagnostics/revision/sparse_runtime_cpu.csv
cd ..
```

Script dùng cùng trọng số nhưng override `k=1,2,3,4`; parameter shapes không đổi theo `k`.

### File đầu ra

```text
diagnostics/revision/sparse_runtime_cpu.csv
```

Báo `mean_ms`, `median_ms`, `p95_ms`, `std_ms` cho từng `k`.

### Điều kiện đo Raspberry Pi

Giữ cố định:

- batch size = 1;
- `--threads 1` cho phép so sánh sạch giữa k;
- cùng Pi, cùng power mode, cùng Python/PyTorch build;
- không chạy tác vụ nặng đồng thời;
- warm-up trước khi đo.

Không trộn hai khái niệm:

1. **PyTorch eager dynamic sparse runtime**: expert chỉ được gọi nếu được Top-k chọn;
2. **ONNX hiện tại**: export tính toàn bộ experts trước khi mask, nên là **dense-export deployment runtime**.

Do đó số ONNX cũ vẫn có thể báo, nhưng không dùng nó làm bằng chứng cho compute saving của Top-k.

---

## 8. Bước E — Q4: controlled Top-k ablation [BẮT BUỘC, CÓ TRAIN]

Đây là run nặng nhất. Giữ cố định `E=4`, Linear context-aware gate và chỉ đổi:

```text
k = 1, 2, 3, 4
seeds = 42, 43, 44, 45, 46
```

### Linux/WSL — chạy toàn bộ

Từ repository root:

```bash
bash scripts/revision/run_q4_topk.sh
```

Script sẽ train đủ 20 cấu hình và sau đó tự đánh giá.

### Chạy thử smoke test trước

Nên chạy một seed, một k trước:

```bash
cd src
python -m revision.train_moe_controlled \
  --dataset plantdoc \
  --num_experts 4 \
  --top_k 1 \
  --seed 42 \
  --epochs 3
cd ..
```

Nếu smoke test thành công, xóa checkpoint smoke-test khỏi nhánh revision trước khi chạy chính thức để không nhầm kết quả.

### Windows hoặc chạy từng cấu hình

Ví dụ:

```powershell
cd src
python -m revision.train_moe_controlled --dataset plantdoc --num_experts 4 --top_k 1 --seed 42
```

Thay `top_k` và `seed` lần lượt. Sau khi đủ 20 runs:

```powershell
python -m revision.evaluate_topk --dataset plantdoc
cd ..
```

### Checkpoint và CSV

```text
checkpoints/plantdoc/revision_topk_linear/mobilenetv3small_moe/4_experts/
├── top_1/seed_42...46/
├── top_2/seed_42...46/
├── top_3/seed_42...46/
└── top_4/seed_42...46/

diagnostics/revision/topk/
├── topk_seed_metrics.csv
└── topk_summary.csv
```

Bảng revision phải lấy từ `topk_summary.csv`, không dùng `results/moe_contextaware_temp0.5.csv` cũ vì file đó thuộc gate/configuration khác.

### Mất điện hoặc dừng giữa chừng

Wrapper mới tạo file `DONE` **chỉ khi process training kết thúc bình thường**. Nếu máy dừng giữa một seed, lần chạy lại sẽ không SKIP seed đó mà tạo một `run_*` mới. Đây là restart an toàn, chưa phải epoch-level resume.

Không tự tạo `DONE`. Nếu muốn ép chạy lại một cấu hình đã hoàn thành, xóa đúng file `DONE` của seed đó; không cần xóa các run cũ. Hàm đánh giá chọn `run_*` mới nhất.

---

## 9. Bước F — Q8: static-uniform 4-expert control [BẮT BUỘC, CÓ TRAIN]

Control này giữ cùng backbone, bốn expert `576 -> 1024 -> 576`, residual path và final classifier nhưng bỏ input-dependent gate. Đây là đối chứng sạch hơn các baseline “parameter-matched” cũ.

Linux/WSL:

```bash
bash scripts/revision/run_q8_static_control.sh
```

Windows/từng seed:

```powershell
cd src
python -m revision.train_static_uniform_control --dataset plantdoc --seed 42
# lặp seeds 43,44,45,46
python -m revision.evaluate_revision_controls
cd ..
```

### Đầu ra chính

```text
checkpoints/plantdoc/revision_controls/static_uniform_4expert/

diagnostics/revision/controls/
├── control_seed_metrics.csv
├── control_summary.csv
└── control_paired_statistics.csv
```

Đối với Q8, ưu tiên so sánh:

```text
Main MoE vs Static-uniform 4-expert
```

trên cả Accuracy và Macro-F1, cùng 5 seed ghép cặp.

Nếu Static-uniform gần hoặc vượt MoE, không được tiếp tục claim rằng routing là nguồn chính của gain. Nếu MoE tốt hơn ổn định, có bằng chứng mạnh hơn rằng gain không chỉ đến từ stored parameter count.

---

## 10. Bước G — kiểm tra ảnh hưởng của 6-D context [ƯU TIÊN CAO, CÓ TRAIN]

Implementation chính dùng sáu image-derived context features, trong khi manuscript cũ chưa mô tả nhánh này. Run này giữ nguyên architecture và parameter count nhưng đặt context về 0 trong cả train/validation/test.

Linux/WSL:

```bash
bash scripts/revision/run_zero_context.sh
```

Windows/từng seed:

```powershell
cd src
python -m revision.train_zero_context_control --dataset plantdoc --seed 42
# lặp 43..46
python -m revision.evaluate_zero_context
cd ..
```

### Đầu ra

```text
checkpoints/plantdoc/revision_controls/moe_zero_context/4_experts/top_2/

diagnostics/revision/context_ablation/
├── context_seed_metrics.csv
└── context_paired_statistics.csv
```

Cách diễn giải:

- nếu main context-aware > zero-context rõ và nhất quán: phải mô tả nhánh 6-D context là một phần thực của phương pháp;
- nếu gần bằng nhau: context path không phải nguồn chính của gain, nhưng manuscript vẫn phải mô tả đúng implementation nếu tiếp tục dùng model này;
- không được xóa context khỏi mô tả trong khi checkpoint chính vẫn sử dụng context-aware gate.

Bản code revision đã sửa trainer để zero-context tensor luôn được chuyển sang đúng device; vì vậy run này chạy được trên CUDA.

---

## 11. Bước H — matched MobileNetV3-Small baseline [TÙY CHỌN NHƯNG NÊN CÓ]

Mục đích là có baseline MobileNetV3-Small cùng implementation family với backbone proposed model.

```bash
bash scripts/revision/run_matched_baseline.sh
```

Hoặc từng seed:

```bash
cd src
python -m revision.train_matched_backbone_baseline --dataset plantdoc --seed 42
# lặp 43..46
python -m revision.evaluate_revision_controls
cd ..
```

Kết quả được đưa chung vào:

```text
diagnostics/revision/controls/
```

Đây là baseline bổ trợ; không cần thêm EfficientNet/GhostNet/ShuffleNet/SqueezeNet mới cho Q2 nếu các kết quả cũ đã hợp lệ.

---

## 12. Nếu muốn chạy trên SLIF

Các Python script hỗ trợ:

```text
--dataset slif
```

nhưng các `.sh` được cung cấp đang hard-code PlantDoc để tránh vô tình trộn hai dataset. Với SLIF nên chạy lệnh Python trực tiếp và đổi output root/dataset tương ứng.

Ví dụ Q4 một cấu hình:

```bash
cd src
python -m revision.train_moe_controlled --dataset slif --num_experts 4 --top_k 2 --seed 42
```

Không bắt buộc lặp toàn bộ revision trên SLIF để trả lời reviewer. Chỉ chạy nếu muốn chứng minh ablation nhất quán trên cả hai dataset và có đủ GPU budget.

---

## 13. Bộ file cuối cùng cần gửi lại để tổng hợp manuscript/response

Sau khi hoàn thành, nén **chỉ các output sau** là đủ để audit kết quả:

```text
diagnostics/revision/model_audit.csv

diagnostics/revision/routing_usage/*.csv

diagnostics/revision/routing_counterfactual/*.csv

diagnostics/revision/sparse_runtime_cpu.csv

diagnostics/revision/topk/*.csv

diagnostics/revision/controls/*.csv

diagnostics/revision/context_ablation/*.csv
```

Kèm theo toàn bộ `training.log` của các run mới:

```text
checkpoints/plantdoc/revision_topk_linear/**/training.log
checkpoints/plantdoc/revision_controls/**/training.log
```

Không cần gửi lại toàn bộ ảnh dataset hoặc toàn bộ checkpoint nếu chỉ cần đánh giá số liệu. Nếu cần kiểm tra architecture/checkpoint sâu hơn, gửi thêm một `best_checkpoint.pth` đại diện cho mỗi loại model.

---

## 14. Tiêu chí quyết định sau khi chạy

### Q1

Nếu runtime tăng tương đối theo `k`, có bằng chứng rằng PyTorch implementation thực sự khai thác dynamic sparse execution. Tuy nhiên vẫn phải báo rõ stored parameters tăng và ONNX hiện tại là dense export.

### Q4

Top-2 chỉ nên được bảo vệ như lựa chọn thực nghiệm nếu `k=2` cho operating point hợp lý về Accuracy/Macro-F1 so với `k=1,3,4`. Nếu `k=3` hoặc `k=4` tốt hơn rõ mà runtime trade-off nhỏ, phải sửa lập luận lựa chọn Top-2 thay vì ép kết luận cũ.

### Q7

Global share tương đối cân bằng nhưng class activation không đều là hoàn toàn có thể chấp nhận: load-balancing loss kiểm soát global utilization, không bắt từng class dùng mọi expert như nhau.

### Q8

Kết luận mạnh nhất chỉ được dùng nếu đồng thời thấy:

1. Main MoE > Static-uniform 4-expert trên paired seeds; và
2. learned routing > shuffled routing.

Khi đó có thể nói input-dependent routing contributes beyond comparable stored expert capacity. Vẫn không gọi đó là bằng chứng trực tiếp về semantic expert specialization.

### Context path

Kết quả zero-context quyết định cách định vị phương pháp. Dù outcome nào, manuscript phải đồng bộ với code và công thức gate phải bao gồm context path nếu main checkpoint dùng `ContextAwareLinearGating`.

---

## 15. Lệnh tối thiểu tóm tắt

Sau khi dataset + main checkpoints đã đặt đúng chỗ:

```bash
# Không train: Q7
cd src
python -m revision.routing_utilization_audit --dataset plantdoc --checkpoint_root ../checkpoints/plantdoc/moe_linearcontextaware_temp0.5/mobilenetv3small_moe/4_experts/top_2

# Không train: Q8 shuffled routes
python -m revision.routing_counterfactual --dataset plantdoc --checkpoint_root ../checkpoints/plantdoc/moe_linearcontextaware_temp0.5/mobilenetv3small_moe/4_experts/top_2 --permutations 100
cd ..

# Có train: Q4
bash scripts/revision/run_q4_topk.sh

# Có train: Q8 matched-capacity control
bash scripts/revision/run_q8_static_control.sh

# Có train: context sensitivity
bash scripts/revision/run_zero_context.sh

# Tùy chọn: matched MNV3 baseline
bash scripts/revision/run_matched_baseline.sh
```

Sau cùng, chạy Q1 sparse runtime trên Raspberry Pi 5 bằng `revision.sparse_runtime_benchmark` và gom các CSV theo Mục 13.
