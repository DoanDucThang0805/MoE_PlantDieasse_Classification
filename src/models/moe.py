from typing import Tuple
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# =========================
# Reproducibility
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# Backbone Feature Extractor
# =========================
class MobileNetV3FeatureExtractor(nn.Module):
    """
    Extract feature vector from pretrained MobileNetV3-Small
    Output dim = 576
    """
    def __init__(self, pretrained=True, freeze_backbone=False) -> None:
        super().__init__()

        backbone = models.mobilenet_v3_small(pretrained=pretrained)

        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output_dim = 576

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x


class NoisyTopkRouter(nn.Module):
    """
    NoisyTopK Router - Một routing mechanism cho Mixture of Experts (MoE)
    Dùng noise trong quá trình training để khuyến khích diversity giữa các experts
    và chỉ select top-k experts có logit cao nhất để giảm computational cost
    """
    def __init__(self, d_model: int, num_experts: int, topk: int) -> None:
        """
        Khởi tạo NoisyTopkRouter
        
        Args:
            d_model: Kích thước của input embedding (số chiều của feature vector)
            num_experts: Tổng số experts có sẵn trong mô hình
            topk: Số lượng experts hàng đầu được chọn (k trong TopK)
        """
        super().__init__()
        # Lưu lại các tham số
        self.d_model = d_model
        self.num_experts = num_experts
        self.topk = topk
        
        # Linear layer để tính routing logits (gate scores) cho mỗi expert
        # Input: d_model chiều, Output: num_experts chiều (một score cho mỗi expert)
        self.gate_linear = nn.Linear(d_model, num_experts)
        
        # Linear layer để tính noise std deviation cho quá trình training
        # Dùng để thêm Gaussian noise vào logits, giúp mô hình tránh overfitting vào một số ít experts
        self.noise_linear = nn.Linear(d_model, num_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass của router
        
        Args:
            x: Input tensor với shape (batch_size, d_model)
            
        Returns:
            router_output: Sparse routing weights với shape (batch_size, num_experts)
                          Chỉ có topk entries khác 0, các entry khác là 0
            indices: Indices của topk experts được chọn, shape (batch_size, topk)
        """
        # ========== Bước 1: Tính routing logits (gate scores) ==========
        # x shape: (batch_size, d_model)
        # gate_linear(x) shape: (batch_size, num_experts)
        # logits[i, j] = score của expert j cho sample i
        logits = self.gate_linear(x)

        # ========== Bước 2: Thêm noise vào logits (chỉ trong training) ==========
        # Noise được dùng để:
        # - Khuyến khích diversity: các experts được chọn có đa dạng hơn
        # - Tránh overfitting: không luôn chọn những experts giống nhau
        # - Improve generalization: mô hình học cách sử dụng nhiều experts
        if self.training:
            # Tính noise standard deviation từ các input
            # noise_linear(x) shape: (batch_size, num_experts)
            noise = self.noise_linear(x)
            
            # F.softplus = log(1 + exp(x))
            # Dùng softplus để đảm bảo noise_std luôn dương
            # noise_std shape: (batch_size, num_experts)
            noise_std = F.softplus(noise)
            
            # Tạo Gaussian noise: torch.randn_like(logits) ~ N(0, 1)
            # torch.randn_like(logits) shape: (batch_size, num_experts)
            # Nhân với noise_std để scale noise: noise ~ N(0, noise_std^2)
            # Cộng vào logits để tạo noisy_logit
            # noisy_logit shape: (batch_size, num_experts)
            noisy_logit = logits + (torch.randn_like(logits) * noise_std)
        else:
            # Ở inference time (testing mode), không thêm noise
            # Chỉ dùng pure logits từ gate_linear
            noisy_logit = logits
        
        # ========== Bước 3: Chọn top-k experts có logit cao nhất ==========
        # torch.topk() trả về: (values, indices) của k biggest elements
        # topk_logits shape: (batch_size, topk) - giá trị logits của topk experts
        # indices shape: (batch_size, topk) - chỉ số của topk experts
        # dim=-1 nghĩa là tìm top-k trên chiều cuối (num_experts dimension)
        topk_logits, indices = torch.topk(noisy_logit, self.topk, dim=-1)

        # ========== Bước 4: Tạo sparse routing mask ==========
        # Khởi tạo một sparse tensor có kích thước giống noisy_logit
        # Tất cả phần tử được set thành -inf (negative infinity)
        # -inf sẽ cho softmax output = 0
        # zeros shape: (batch_size, num_experts)
        zeros = torch.full_like(noisy_logit, float('-inf'))
        
        # scatter(): Đặt các logits của topk experts vào đúng vị trí
        # Cú pháp: tensor.scatter(dim, indices, src)
        # - dim=-1: scatter trên chiều cuối
        # - indices: vị trí các topk experts (shape: batch_size, topk)
        # - topk_logits: giá trị logits cần đặt (shape: batch_size, topk)
        # Result: sparse_logits là một tensor (batch_size, num_experts)
        #         chỉ có topk entries từ topk_logits, các entries khác là -inf
        sparse_logits = zeros.scatter(-1, indices, topk_logits)
        
        # ========== Bước 5: Áp dụng softmax để convert logits thành weights ==========
        # F.softmax(x, dim=-1) = exp(x) / sum(exp(x)) trên từng row
        # Vì sparse_logits có -inf entries, exp(-inf) = 0
        # Nên softmax sẽ cho 0 cho tất cả non-topk entries
        # Và normalize topk entries để sum = 1
        # router_output shape: (batch_size, num_experts)
        # router_output[i, j] = 0 nếu expert j không trong topk
        # router_output[i, j] > 0 nếu expert j trong topk
        # sum(router_output[i, :]) = 1 (valid probability distribution)
        router_output = F.softmax(sparse_logits, dim=-1)
        
        # ========== Trả về kết quả ==========
        # router_output: Sparse routing weights dùng để combine outputs từ experts
        # indices: Expert indices dùng để tìm outputs của từng expert
        return router_output, indices
    

class Expert(nn.Module):
    def __init__(self, d_model: int, hiddent_scale: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model*hiddent_scale),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model*hiddent_scale, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    

class MoeLayer(nn.Module):
    def __init__(self, d_model: int, num_classes: int, num_experts: int, hiddent_scale: int, topk: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.topk = topk
        self.num_classes = num_classes

        self.gate = NoisyTopkRouter(d_model, num_experts, topk)
        self.experts = nn.ModuleList(
            [Expert(d_model, hiddent_scale, num_classes) for _ in range(num_experts)]
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass của Mixture of Experts layer
        
        Quy trình:
        1. Router chọn top-k experts phù hợp cho mỗi sample
        2. Gửi mỗi sample đến những experts mà nó được route tới
        3. Kết hợp outputs từ các experts bằng weighted gates từ router
        
        Args:
            x: Input tensor shape (batch_size, d_model)
        
        Returns:
            final_output: Output dự đoán từ MoE layer, shape (batch_size, num_classes)
            router_output: Sparse routing weights, shape (batch_size, num_experts)
            topk_indices: Indices của topk experts được chọn, shape (batch_size, topk)
        """
        # ========== Lấy thông tin từ input ==========
        # Lấy batch_size để khởi tạo output tensor
        # x shape: (batch_size, d_model)
        batch_size, _ = x.shape
        
        # ========== Gọi router để xác định experts ==========
        # router_output shape: (batch_size, num_experts) - sparse weights cho mỗi expert
        # topk_indices shape: (batch_size, topk) - indices của topk experts được chọn
        # Mỗi sample được route tới topk experts (không phải tất cả num_experts)
        router_output, topk_indices = self.gate(x)
        
        # ========== Khởi tạo output tensor ==========
        # Tạo tensor zeros để tích lũy outputs từ các experts
        # final_output shape: (batch_size, num_classes)
        # TỐI ƯU: Di chuyển tensor sang device giống x (CPU hoặc GPU)
        final_output = torch.zeros((batch_size, self.num_classes), device=x.device)
        
        # ========== Lặp qua từng expert ==========
        # Vì router chỉ chọn topk experts, nên nhiều experts không được sử dụng
        # Nhưng ta vẫn lặp qua tất cả để đơn giản hóa logic
        for expert_ith in range(self.num_experts):
            # ========== Tìm samples nào được route tới expert này ==========
            # topk_indices shape: (batch_size, topk)
            # topk_indices[i, j] = index của j-th expert trong top-k của sample i
            # (topk_indices == expert_ith) tạo boolean mask: (batch_size, topk)
            #   - True nơi expert_ith được route
            #   - False nơi khác
            embedding_mask = (topk_indices == expert_ith)
            
            # ========== Tối ưu: Tính mask một lần thôi ==========
            # embedding_mask.any(dim=1) shape: (batch_size,)
            # Trả về True nếu sample i được route tới expert_ith
            # TỐI ƯU: Lưu vào biến để tránh tính lại nhiều lần
            sample_mask = embedding_mask.any(dim=1)
            
            # ========== Skip nếu expert này không phục vụ sample nào ==========
            # sample_mask.sum() = số samples được route tới expert này
            # Nếu = 0, không cần xử lý expert này
            if sample_mask.sum() == 0:
                continue
            
            # ========== Lấy embeddings của samples được route tới expert này ==========
            # sample_mask shape: (batch_size,)
            # x[sample_mask] shape: (num_selected_samples, d_model)
            # Chỉ lấy những samples được route tới expert này
            selected_embeddings = x[sample_mask]
            
            # ========== Gửi embeddings qua expert này ==========
            # self.experts[expert_ith] là một Expert(d_model, scale, num_classes) module
            # Input: (num_selected_samples, d_model)
            # Output: (num_selected_samples, num_classes) - prediction của expert cho samples này
            expert_outputs = self.experts[expert_ith](selected_embeddings)
            
            # ========== Lấy routing weights cho expert này ==========
            # router_output shape: (batch_size, num_experts)
            # router_output[:, expert_ith] shape: (batch_size,)
            #   - weight của expert_ith cho mỗi sample
            # [sample_mask] shape: (num_selected_samples,)
            #   - lọc ra chỉ weights cho samples được route
            #   - Lưu ý: các samples khác có weight = 0 ở expert này
            weighted_gates = router_output[:, expert_ith][sample_mask]
            
            # ========== Cộng weighted expert outputs vào final output ==========
            # expert_outputs shape: (num_selected_samples, num_classes)
            # weighted_gates shape: (num_selected_samples,)
            # weighted_gates.unsqueeze(1) shape: (num_selected_samples, 1)
            #   - unsqueeze(1) để broadcast multiply: (num_selected_samples, 1) * (num_selected_samples, num_classes)
            # Kết quả: (num_selected_samples, num_classes) - weighted expert predictions
            weighted_expert_output = expert_outputs * weighted_gates.unsqueeze(1)
            
            # final_output[sample_mask] += weighted_expert_output
            # Cộng weighted expert output vào final output cho những samples được route
            # Note: Nếu sample được route tới nhiều experts, outputs từ các experts sẽ được cộng vào
            # Sau softmax từ router, tổng weights của một sample = 1, nên tổng final output hợp lý
            final_output[sample_mask] += weighted_expert_output
        
        # ========== Trả về kết quả ==========
        # final_output shape: (batch_size, num_classes)
        #   - Kết quả dự đoán kết hợp từ các experts cho mỗi sample
        # router_output shape: (batch_size, num_experts)
        #   - Sparse routing weights: chỉ topk entries khác 0, còn lại = 0
        #   - Dùng để debug, visualize routing decisions
        # topk_indices shape: (batch_size, topk)
        #   - Indices của topk experts được chọn cho mỗi sample
        #   - Dùng để debug, phân tích expert utilization
        return final_output, router_output, topk_indices


class MobilenetMoE(nn.Module):
    """
    MobileNet + Mixture of Experts (MoE) model
    
    Quy trình:
    1. MobileNetV3-Small trích xuất features từ ảnh input
    2. MoE layer route features tới multiple experts và kết hợp outputs
    
    Lợi ích:
    - MobileNet: Lightweight backbone, phù hợp cho inference trên device
    - MoE: Tăng model capacity mà không tăng computational cost quá nhiều
      (vì chỉ dùng topk experts thay vì tất cả)
    """
    def __init__(
        self,
        num_classes: int,
        hiddent_scale: int,
        num_experts: int,
        topk: int,
        pretrained_backbone: bool=True,
        freeze_backbone: bool = False
    ) -> None:
        """
        Khởi tạo MobileNet + MoE model
        
        Args:
            num_classes: Số lượng output classes
            hiddent_scale: Scale factor cho hidden layer size trong experts
            num_experts: Tổng số experts
            topk: Số lượng experts được route cho mỗi sample
            pretrained_backbone: Dùng pretrained weights cho MobileNetV3
            freeze_backbone: Freeze backbone weights (không update khi training)
        """
        super().__init__()
        self.feature_extractor = MobileNetV3FeatureExtractor(pretrained_backbone, freeze_backbone)
        self.moe_layer = MoeLayer(
            d_model= self.feature_extractor.output_dim,
            num_classes=num_classes,
            num_experts=num_experts,
            hiddent_scale=hiddent_scale,
            topk=topk
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass của MobileNet + MoE model
        
        Args:
            x: Input image tensor shape (batch_size, 3, 224, 224)
        
        Returns:
            logits: Kết quả dự đoán shape (batch_size, num_classes)
            router_output: Routing weights shape (batch_size, num_experts)
            topk_indices: Top-k expert indices shape (batch_size, topk)
        """
        # Trích xuất features từ MobileNetV3
        # feature_inputs shape: (batch_size, 576)
        feature_inputs = self.feature_extractor(x)
        
        # Đưa qua MoE layer để kết hợp predictions từ multiple experts
        # Trả về 3 giá trị: predictions, routing weights, expert indices
        logits, router_output, topk_indices = self.moe_layer(feature_inputs)
        return logits, router_output, topk_indices
        


model = MobilenetMoE(
    num_classes=10,
    hiddent_scale=2,
    num_experts=3,
    topk=2,
    pretrained_backbone=True,
    freeze_backbone=False
)


