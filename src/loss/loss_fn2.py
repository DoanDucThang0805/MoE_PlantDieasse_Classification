"""
MoELossWithOrtho – Hàm mất mát mở rộng cho Mixture of Experts.

So với MoELoss gốc (loss_fn.py), file này bổ sung thêm:
    - Orthogonal Regularization Loss (L_orthogonal)

Tổng loss:
    L_total = L_class + alpha * L_balance + lambda_ortho * L_orthogonal

Trong đó:
    L_class      : CrossEntropy phân loại chính                (giữ nguyên từ bản gốc)
    L_balance    : Load-balance loss, đảm bảo các expert được  (giữ nguyên từ bản gốc)
                   sử dụng đều đặn
    L_orthogonal : Khuyến khích các expert học những feature    (MỚI – theo paper Salman et al.)
                   không trùng lặp nhau

Tham chiếu công thức (paper Salman et al., 2025 – mục 3.2):
    L_orthogonal = Σ_{i=1}^{K} Σ_{j=i+1}^{K}  ‖ W_i @ W_j^T ‖_F

    - W_i, W_j  : ma trận trọng số của expert thứ i và j
    - ‖·‖_F    : chuẩn Frobenius
    - Gradient chảy qua tất cả các W, buộc các expert học
      feature orthogonal (không overlap) với nhau.

Cách dùng trong moe_train.py
─────────────────────────────
    from loss.loss_fn_v2 import MoELossWithOrtho

    criterion = MoELossWithOrtho(
        alpha        = 0.01,          # weight của L_balance  (giữ nguyên)
        lambda_ortho = 0.001,         # weight của L_orthogonal (chỉnh tuỳ dataset)
        class_weights= class_weights, # optional – xử lý class imbalance
    )

    # Gọi trong training loop – truyền thêm model.moe_layer.experts
    loss = criterion(
        logits,
        labels,
        clean_logits,
        topk_indices,
        experts = model.moe_layer.experts,   # nn.ModuleList
    )

Ghi chú backward-compatibility
────────────────────────────────
    - Tham số `experts` là Optional → nếu không truyền vào, L_orthogonal = 0
      và hành vi hoàn toàn giống MoELoss gốc.
    - Signature forward() tương thích với trainer hiện tại nếu dùng keyword arg.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELossWithOrtho(nn.Module):
    """
    Hàm mất mát MoE mở rộng với Orthogonal Regularization.

    Parameters
    ----------
    alpha : float
        Trọng số của L_balance (load-balance auxiliary loss).
        Mặc định: 0.01.
    lambda_ortho : float
        Trọng số của L_orthogonal.
        - Nếu = 0  → tắt hoàn toàn, giống MoELoss gốc.
        - Khuyến nghị: bắt đầu từ 0.001 rồi tune bằng Optuna.
        Mặc định: 0.001.
    class_weights : torch.Tensor, optional
        Trọng số từng class để bù class imbalance (truyền vào CrossEntropyLoss).
    """

    def __init__(
        self,
        alpha: float = 0.01,
        lambda_ortho: float = 0.001,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.alpha = alpha
        self.lambda_ortho = lambda_ortho

        # ── CrossEntropy Loss (có hỗ trợ class weights) ──────────────────────
        if class_weights is not None:
            self.ce = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.ce = nn.CrossEntropyLoss()

    # ─────────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        router_logits: torch.Tensor,
        top_k_indices: torch.Tensor,
        experts: Optional[nn.ModuleList] = None,
    ) -> torch.Tensor:
        """
        Tính tổng mất mát cho MoE.

        Parameters
        ----------
        logits : Tensor  [B, num_classes]
            Output phân loại của mô hình.
        targets : Tensor  [B]
            Nhãn thực tế.
        router_logits : Tensor  [B, N]
            Clean logits từ gating network (N = số expert).
        top_k_indices : Tensor  [B, K]
            Chỉ số K expert được chọn cho từng mẫu.
        experts : nn.ModuleList, optional
            Danh sách các expert network từ MoELayer.
            Truyền vào để tính L_orthogonal.
            Nếu None hoặc lambda_ortho == 0 → bỏ qua.

        Returns
        -------
        Tensor (scalar)
            L_total = L_class + alpha * L_balance + lambda_ortho * L_orthogonal
        """

        # ── 1. Classification loss ────────────────────────────────────────────
        task_loss = self.ce(logits, targets)

        # ── 2. Load-balance auxiliary loss ────────────────────────────────────
        balance_loss = self._compute_load_balance_loss(router_logits, top_k_indices)

        # ── 3. Orthogonal regularization loss ─────────────────────────────────
        if experts is not None and self.lambda_ortho > 0.0:
            ortho_loss = self._compute_orthogonal_loss(experts)
        else:
            ortho_loss = torch.tensor(0.0, device=logits.device)

        # ── 4. Tổng hợp ───────────────────────────────────────────────────────
        total_loss = (
            task_loss
            + self.alpha * balance_loss
            + self.lambda_ortho * ortho_loss
        )

        return total_loss

    # ─────────────────────────────────────────────────────────────────────────
    # Load-balance Loss  (giữ nguyên từ MoELoss gốc)
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_load_balance_loss(
        self,
        router_logits: torch.Tensor,
        top_k_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Tính auxiliary loss để cân bằng tải giữa các expert.

        Công thức:  L_balance = N * Σ_i (f_i * P_i)

            P_i : xác suất trung bình router chọn expert i (từ softmax)
            f_i : tỷ lệ thực tế expert i được chọn trong batch

        Gradient chỉ chảy qua P_i → ép router cân bằng hơn.

        Parameters
        ----------
        router_logits  : [B, N]
        top_k_indices  : [B, K]

        Returns
        -------
        Tensor (scalar)
        """
        _, num_experts = router_logits.shape
        _, top_k = top_k_indices.shape

        # P_i – xác suất trung bình mỗi expert được chọn
        router_probs = torch.softmax(router_logits, dim=-1)   # [B, N]
        P_i = router_probs.mean(dim=0)                        # [N]

        # f_i – tỷ lệ mẫu thực sự được gán cho mỗi expert
        expert_one_hot = F.one_hot(
            top_k_indices, num_classes=num_experts
        )                                                      # [B, K, N]
        expert_selection_count = expert_one_hot.sum(dim=1).float()  # [B, N]
        f_i = expert_selection_count.mean(dim=0) / top_k     # [N]

        load_balance_loss = num_experts * torch.sum(f_i * P_i)
        return load_balance_loss

    # ─────────────────────────────────────────────────────────────────────────
    # Orthogonal Regularization Loss  (MỚI)
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_orthogonal_loss(self, experts: nn.ModuleList) -> torch.Tensor:
        """
        Tính orthogonal regularization loss giữa các expert.

        Công thức (Salman et al., 2025 – mục 3.2):
            L_orthogonal = Σ_{i=1}^{K} Σ_{j=i+1}^{K}  ‖ W_i @ W_j^T ‖_F

        Trong đó W_i là ma trận trọng số của Linear layer đầu tiên của expert i.
        Layer đầu tiên được chọn vì nó chiếu feature đầu vào lên không gian ẩn
        → là nơi thể hiện rõ nhất tính đặc thù (specialization) của mỗi expert.

        Nếu W_i và W_j orthogonal hoàn toàn:  W_i @ W_j^T = 0  → loss = 0
        Nếu chúng giống nhau:                 ‖W_i @ W_j^T‖_F lớn → bị phạt

        Parameters
        ----------
        experts : nn.ModuleList
            Danh sách các expert network từ MoELayer.moe_layer.experts.
            Mỗi expert là nn.Sequential: [Linear, LayerNorm, GELU, Dropout, Linear]

        Returns
        -------
        Tensor (scalar)
            Tổng chuẩn Frobenius của tất cả các cặp expert, đã normalize
            theo số cặp để giữ giá trị loss ổn định khi thay đổi num_experts.
        """
        ortho_loss = torch.tensor(0.0, device=next(experts[0].parameters()).device)
        num_experts = len(experts)
        num_pairs = 0

        for i in range(num_experts):
            for j in range(i + 1, num_experts):

                # Lấy weight của Linear layer đầu tiên của mỗi expert
                # experts[i][0] = nn.Linear(embedding_dim, 1024)
                # weight shape: [out_features, in_features] = [1024, embedding_dim]
                Wi = experts[i][0].weight   # [1024, D]
                Wj = experts[j][0].weight   # [1024, D]

                # Tính W_i @ W_j^T  →  shape: [1024, 1024]
                # Sau đó lấy chuẩn Frobenius để đo mức độ overlap
                gram_matrix = Wi @ Wj.T                         # [1024, 1024]
                pair_loss = torch.norm(gram_matrix, p="fro")    # scalar

                ortho_loss = ortho_loss + pair_loss
                num_pairs += 1

        # Normalize theo số cặp để loss không bùng nổ khi num_experts lớn
        if num_pairs > 0:
            ortho_loss = ortho_loss / num_pairs

        return ortho_loss