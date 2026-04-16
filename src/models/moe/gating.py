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
    Bộ gating Top-K có bổ sung nhiễu cho Mixture of Experts (MoE).

    Cơ chế hoạt động:
    - Tính toán logits gating cơ bản (clean logits)
    - Trong quá trình huấn luyện (training), thêm nhiễu Gaussian vào logits
    - Chọn Top-K chuyên gia từ logits có nhiễu
    - Áp dụng softmax với temperature để chuẩn hóa trọng số

    Lợi ích:
    - Nhiễu giúp tăng tính đa dạng trong việc lựa chọn chuyên gia
    - Tránh hiện tượng một số expert luôn bị bỏ qua
    - Temperature điều chỉnh độ sắc của phân phối gating:
        + temperature thấp → phân phối sắc → tăng chuyên môn hóa
        + temperature cao → phân phối phẳng → tăng chia tải giữa các expert
    """
    
    def __init__(self, model_dim: int, num_experts: int, top_k: int, temperature: float = 1.0, noise_stddev: float = 1.0):
        """
        Khởi tạo bộ gating Top-K có bổ sung nhiễu.

        Tham số:
        -----------
        model_dim : int
            Kích thước embedding/feature input (ví dụ: 960)

        num_experts : int
            Tổng số chuyên gia có sẵn (ví dụ: 4)

        top_k : int
            Số lượng chuyên gia hàng đầu được chọn (ví dụ: 2)

        temperature : float, tùy chọn
            Hệ số temperature dùng trong softmax của Top-K logits.
            Điều chỉnh độ sắc của phân phối gating:
            - temperature < 1 → phân phối sắc hơn (tăng specialization)
            - temperature > 1 → phân phối phẳng hơn (tăng load balancing)

        noise_stddev : float, tùy chọn
            Độ lệch chuẩn của nhiễu Gaussian được thêm vào logits trong training.
        Điều khiển cường độ nhiễu để khuyến khích khám phá các expert khác nhau.
        """
        
        super().__init__()
        self.model_dim = model_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_stddev = noise_stddev
        self.temperature = temperature

        # Lớp tuyến tính để dự đoán logits gating cơ bản
        self.gate_projector = nn.Sequential(
            nn.Linear(self.model_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, self.num_experts, bias=False)
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
        4. Chuẩn hóa Top-K logits bằng softmax có temperature
        
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
        combined_weights = F.softmax(top_k_logits / self.temperature, dim=-1)

        # Trả về trọng số, chỉ số chuyên gia, và logits gốc sạch
        return combined_weights, top_k_indices, clean_logits    
