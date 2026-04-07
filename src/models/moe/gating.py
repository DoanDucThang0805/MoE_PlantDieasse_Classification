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
        self.gate_projector = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim // 4),  # Giảm chiều để tăng tính phi tuyến
            nn.LayerNorm(self.model_dim // 4),  # Thêm layer norm để ổn định training
            nn.GELU(),

            nn.Linear(self.model_dim//4, self.model_dim//16),  # Giảm tiếp để tăng tính phi tuyến
            nn.LayerNorm(self.model_dim//16),  # Thêm layer norm để ổn định training
            nn.GELU(),

            nn.Linear(self.model_dim//16, self.num_experts, bias=False)
        )
        
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
