Session 5. Chẩn đoán expert routing
Mục tiêu: kiểm tra MoE có thực sự tạo computation có điều kiện theo lớp hay chỉ là một classifier nhiều tham số với expert usage tương đối đều. Expert usage hiện chỉ cho thấy router không collapse; nó chưa chứng minh expert specialization.
Chỉ số cần tính: với lớp c và expert e, class-conditioned expert activation rate là rho_{c,e} = (1 / |I_c|) sum_{i in I_c} I(e in K_i), trong đó I_c là tập mẫu thuộc lớp c. Nếu các lớp khác nhau kích hoạt các expert khác nhau, có thể nói routing có class-conditioned behavior. Nếu mọi hàng gần giống nhau, không claim disease-specific specialization.
Chẩn đoán
Đầu ra cần có
Global expert usage
Đã có cho Linear và MLP Gating
Class-wise routing heatmap
FACT{Tạo heatmap rho_{c,e}}
Routing entropy per class
FACT{Báo cáo entropy trung bình cho từng lớp}
Expert-conditioned confusion matrix
FACT{Tùy chọn nếu đủ mẫu cho từng expert}
Feature visualization theo selected expert
FACT{Tùy chọn t-SNE/UMAP}

Vai trò của session này là giải thích cơ chế, không phải tăng headline metric. Nếu routing heatmap có cấu trúc theo lớp, bài sẽ mạnh hơn rõ rệt.

Session 6. Kiểm định thống kê theo seed
Mục tiêu: kiểm tra khác biệt giữa hai mô hình có ổn định qua 5 seed hay không. Mean ± std là cần thiết nhưng chưa đủ để khẳng định hai mô hình tách biệt thống kê.
Nếu hai mô hình dùng cùng 5 split và seed, dùng paired test. Với metric m, hiệu giữa hai mô hình tại seed s là Delta_s = m_s^(A) - m_s^(B). Paired t-test dùng t = mean(Delta) / (std(Delta) / sqrt(S)). Vì S = 5 là nhỏ, cần báo cáo thêm Wilcoxon signed-rank test nếu có seed-wise values.
So sánh
Đầu ra thống kê cần có
MoE vs MobileNetV3-Small
FACT{p-value cho Accuracy và Macro-F1}
MoE vs widened MLP head
FACT{p-value cho Accuracy và Macro-F1}
MoE vs dense multi-branch head
FACT{p-value cho Accuracy và Macro-F1}
Linear Gating vs MLP Gating
FACT{p-value cho Accuracy và Macro-F1}
MoE vs lightweight backbones
FACT{p-value nếu paired runs có sẵn}

Nếu 5 lần chạy không paired theo cùng split và seed, không được dùng paired test. Khi đó phải rerun toàn bộ theo fixed seed IDs. Không claim statistical superiority khi paired test không kết luận.