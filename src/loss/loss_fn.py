import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELoss(nn.Module):
    """
    Hàm mất mát cho mô hình Mixture of Experts (MoE).
    
    Kết hợp hai phần:
    1. Task loss: Lỗi phân loại chính (Cross Entropy)
    2. Auxiliary loss: Lỗi cân bằng tải để phân phối mẫu đều giữa các expert
    
    Parameters:
        alpha (float): Trọng số của auxiliary loss (mặc định: 0.01)
    """
    
    def __init__(self, alpha: float = 0.01):
        """
        Khởi tạo MoELoss.
        
        Args:
            alpha: Hệ số cân bằng giữa task loss và auxiliary loss
        """
        super().__init__()
        self.alpha = alpha  # Trọng số của auxiliary loss
        self.ce = nn.CrossEntropyLoss()  # Hàm mất mát phân loại chính


    def forward(self, logits, targets, router_logits, top_k_indices):
        """
        Tính tổng mất mát cho Mixture of Experts.
        
        Args:
            logits (Tensor):         Output của mô hình, shape (B, num_classes)
                                     B = batch size, num_classes = số lớp
            targets (Tensor):        Nhãn từng mẫu, shape (B,)
            router_logits (Tensor):  Output của router (định hướng), shape (B, N)
                                     N = số expert
            top_k_indices (Tensor):  Chỉ số K expert được chọn, shape (B, K)
                                     K = số expert được lựa chọn cho từng mẫu
        
        Returns:
            Tensor: Tổng mất mát (task_loss + alpha * auxiliary_loss)
        """
        
        # ========== Tính task loss (mất mát phân loại chính) ==========
        # So sánh output của mô hình với nhãn thực tế
        task_loss = self.ce(logits, targets)

        # ========== Tính auxiliary loss (mất mát cân bằng tải) ==========
        # Đảm bảo các expert được sử dụng đều đặn, tránh tình huống 
        # nhiều mẫu chỉ sử dụng vài expert
        auxiliary_loss = self._calculate_load_balance_loss(router_logits, top_k_indices)

        # ========== Tính tổng mất mát ==========
        # Kết hợp hai mất mát với trọng số alpha
        total_loss = task_loss + self.alpha * auxiliary_loss

        return total_loss


    def _calculate_load_balance_loss(self, router_logits, top_k_indices):
        """
        Tính auxiliary loss để cân bằng tải giữa các expert.
        
        Mục đích: Khuyến khích router phân phối mẫu đều đặn giữa các expert,
        tránh tình huống một vài expert được sử dụng quá nhiều.
        
        Args:
            router_logits (Tensor): Output của router, shape (B, N)
                                    B = batch size, N = số expert
            top_k_indices (Tensor): Chỉ số K expert được chọn, shape (B, K)
                                    K = số expert được lựa chọn
        
        Returns:
            Tensor: Giá trị loss (vô hướng)
        """
        _, num_experts = router_logits.shape
        _, top_k = top_k_indices.shape

        # ===== Bước 1: Tính P_i - Xác suất router chọn expert =====
        # Chuyển router_logits thành xác suất qua Softmax
        router_probs = torch.softmax(router_logits, dim=-1)  # shape: (B, N)
        
        # Tính xác suất trung bình mỗi expert được chọn theo quyết định router
        # P_i[i] = xác suất trung bình expert i được chọn
        P_i = router_probs.mean(dim=0)  # shape: (N,)

        # ===== Bước 2: Tính f_i - Tỷ lệ mẫu được gán cho mỗi expert =====
        # Chuyển chỉ số expert thành dạng one-hot
        expert_one_hot = F.one_hot(top_k_indices, num_classes=num_experts)  # shape: (B, K, N)
        
        # Đếm số lần mỗi expert được chọn trong một batch
        expert_selection_count = expert_one_hot.sum(dim=1).float()  # shape: (B, N)
        
        # Tính tỷ lệ mẫu được gán cho mỗi expert
        # f_i[i] = (số lần expert i được chọn) / (tổng lựa chọn)
        f_i = expert_selection_count.mean(dim=0) / top_k  # shape: (N,)

        # ===== Bước 3: Tính loss =====
        # Loss = N * sum(f_i * P_i)
        # Min = 1 khi perfectly balanced (f_i = P_i = 1/N)
        # Max = N khi collapsed (1 expert nhận tất cả)
        # Gradient chỉ chảy qua P_i để ép router cân bằng
        load_balance_loss = num_experts * torch.sum(f_i * P_i)

        return load_balance_loss
        