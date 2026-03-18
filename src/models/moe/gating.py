"""
Mô-đun Gating cho Mixture of Experts (MoE)

Mô-đun này cung cấp hai cơ chế quản cầu chọn lựa chuyên gia (gating):
1. StandardTopKgating: Bộ gating tiêu chuẩn sử dụng Top-K
2. NoisyTopKGating: Bộ gating có bổ sung nhiễu để tăng sự đa dạng

Tác giả: MoE Team
Ngày tạo: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class StandardTopKgating(nn.Module):
    """
    Bộ gating Top-K tiêu chuẩn cho Mixture of Experts.
    
    Chức năng:
    - Dự đoán trọng số cho từng chuyên gia
    - Chọn Top-K chuyên gia có trọng số cao nhất
    - Áp dụng softmax trên Top-K logits
    """
    def __init__(self, model_dim: int, num_experts: int, top_k: int):
        """
        Khởi tạo bộ gating Top-K tiêu chuẩn.
        
        Tham số:
        -----------
        model_dim : int
            Kích thước embedding/feature input (ví dụ: 960)
        num_experts : int
            Tổng số chuyên gia có sẵn (ví dụ: 4)
        top_k : int
            Số lượng chuyên gia hàng đầu được chọn (ví dụ: 3)
        """
        super().__init__()
        self.model_dim = model_dim
        self.num_experts = num_experts
        self.top_k = top_k
        # Lớp tuyến tính để dự đoán logits cho từng chuyên gia
        self.gate_projector = nn.Linear(self.model_dim, self.num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tính toán trọng số gating cho các chuyên gia.
        
        Quy trình:
        1. Tính logits dự đoán cho tất cả chuyên gia
        2. Chọn Top-K chuyên gia có logits cao nhất
        3. Áp dụng softmax trên Top-K logits để được trọng số chuẩn hóa
        
        Tham số:
        -----------
        x : torch.Tensor
            Tensor input có shape [batch_size, model_dim]
        
        Trả về:
        -----------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - combined_weights: Trọng số softmax của Top-K [batch_size, top_k]
            - top_k_indices: Chỉ số của K chuyên gia được chọn [batch_size, top_k]
            - gate_logits: Logits gốc tất cả chuyên gia [batch_size, num_experts]
        """
        # Tính logits dự đoán cho tất cả chuyên gia
        gate_logits = self.gate_projector(x)
        
        # Chọn Top-K logits và chỉ số tương ứng
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        
        # Chuẩn hóa Top-K logits bằng softmax để được trọng số
        combined_weights = F.softmax(top_k_logits, dim=-1, dtype=torch.float32)
        
        return combined_weights, top_k_indices, gate_logits
    

class NoisyTopKGating(nn.Module):
    """
    Bộ gating Top-K có bổ sung nhiễu cho Mixture of Experts.
    
    Chức năng:
    - Tính toán logits gating cơ bản (clean logits)
    - Trong quá trình huấn luyện (training), thêm nhiễu Gaussian vào logits
    - Chọn Top-K chuyên gia từ logits có nhiễu
    - Áp dụng softmax để chuẩn hóa trọng số
    
    Lợi ích của việc bổ sung nhiễu:
    - Tăng sự đa dạng khi chọn chuyên gia
    - Giảm việc một số chuyên gia luôn bị chọn
    - Cải thiện tính tổng quát của mô hình
    """
    
    def __init__(self, model_dim: int, num_experts: int, top_k: int, noise_stddev=1.0):
        """
        Khởi tạo bộ gating Top-K có bổ sung nhiễu.
        
        Tham số:
        -----------
        model_dim : int
            Kích thước embedding/feature input (ví dụ: 960)
        num_experts : int
            Tổng số chuyên gia có sẵn (ví dụ: 4)
        top_k : int
            Số lượng chuyên gia hàng đầu được chọn (ví dụ: 3)
        noise_stddev : float, tùy chọn
            Độ lệch chuẩn của nhiễu Gaussian, mặc định = 1.0
            Kiểm soát độ mạnh của nhiễu thêm vào
        """
        super().__init__()
        self.model_dim = model_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_stddev = noise_stddev

        # Lớp tuyến tính để dự đoán logits gating cơ bản
        self.gate_projector = nn.Linear(model_dim, num_experts, bias=False)
        
        # Lớp tuyến tính để dự đoán độ lớn của nhiễu cho mỗi chuyên gia
        self.noise_layer = nn.Linear(model_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        """
        Tính toán trọng số gating có bổ sung nhiễu cho các chuyên gia.
        
        Quá trình:
        1. Tính logits gating cơ bản (clean logits)
        2. Nếu đang huấn luyện (training):
           - Tính độ lớn của nhiễu từ noise_layer
           - Tạo nhiễu Gaussian
           - Cộng nhiễu vào clean logits
        3. Chọn Top-K chuyên gia từ logits (có nhiễu hoặc không)
        4. Chuẩn hóa Top-K logits bằng softmax
        
        Tham số:
        -----------
        x : torch.Tensor
            Tensor input có shape [batch_size, model_dim]
        
        Trả về:
        -----------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - combined_weights: Trọng số softmax của Top-K [batch_size, top_k]
            - top_k_indices: Chỉ số của K chuyên gia được chọn [batch_size, top_k]
            - clean_logits: Logits gốc không có nhiễu [batch_size, num_experts]
        """

        # Tính logits gating cơ bản (sạch, chưa có nhiễu)
        clean_logits = self.gate_projector(x)

        # Chỉ thêm nhiễu trong quá trình huấn luyện
        if self.training:
            # Tính độ lớn của nhiễu dựa trên input
            noise_magnitude = self.noise_layer(x)
            
            # Áp dụng softplus để đảm bảo noise_scale là dương
            # softplus(x) = log(1 + exp(x)) giúp tránh scale âm
            noise_scale = torch.nn.functional.softplus(noise_magnitude)

            # Tạo nhiễu Gaussian cùng shape với clean_logits
            sampled_noise = torch.randn_like(clean_logits)

            # Kết hợp clean logits với nhiễu đã được điều chỉnh độ lớn
            noisy_logits = clean_logits + noise_scale * sampled_noise * self.noise_stddev
        else:
            # Nếu không huấn luyện, sử dụng logits sạch mà không có nhiễu
            noisy_logits = clean_logits

        # Chọn Top-K logits cao nhất và chỉ số chuyên gia tương ứng
        top_k_logits, top_k_indices = torch.topk(noisy_logits, self.top_k, dim=-1)

        # Áp dụng softmax trên Top-K logits để chuẩn hóa thành trọng số
        combined_weights = F.softmax(top_k_logits, dim=-1)

        # Trả về trọng số, chỉ số chuyên gia, và logits gốc sạch
        return combined_weights, top_k_indices, clean_logits    


if __name__ == "__main__":
    """
    Khối kiểm tra (test block) để xác minh hoạt động của gating modules.
    
    Khởi tạo một bộ gating StandardTopKgating và kiểm tra output.
    """
    
    # Khởi tạo bộ gating với các tham số:
    # - model_dim=960: Kích thước feature input
    # - num_experts=4: Tổng 4 chuyên gia
    # - top_k=3: Chọn 3 chuyên gia tốt nhất
    noisygating = StandardTopKgating(
        model_dim=960,
        num_experts=4,
        top_k=1
    )
    
    # Tạo dữ liệu test: batch_size=3, model_dim=960
    logits = torch.rand((3, 960))
    
    # Chạy forward pass
    combined_weights, top_k_indices, clean_logits = noisygating(logits)
    
    # In kết quả
    print(combined_weights)  # Trọng số trơn mượt từ softmax
    print(top_k_indices)      # Chỉ số của 3 chuyên gia được chọn
    print(clean_logits)      # Logits gốc cho tất cả chuyên gia (4)
