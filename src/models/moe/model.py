import torch
import torch.nn as nn
from backbone import Mobilenetv3LargeFeatureExtractor
from gating import NoisyTopKGating
import warnings

warnings.filterwarnings("ignore")


class MoELayer(nn.Module):
    """
    Lớp Mixture of Experts (MoE) - cho phép mô hình sử dụng nhiều chuyên gia (experts)
    và gating network để định tuyến dữ liệu đến các chuyên gia phù hợp.
    
    Args:
        model_dim (int): Kích thước của feature vector
        num_experts (int): Số lượng chuyên gia
        top_k (int): Số lượng chuyên gia hàng đầu được lựa chọn cho mỗi input
    """
    
    def __init__(self, model_dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Gating network: quyết định chuyên gia nào xử lý từng mẫu
        self.gating = NoisyTopKGating(
            model_dim=model_dim, 
            num_experts=num_experts, 
            top_k=top_k
        )
        
        # Danh sách các chuyên gia - mỗi chuyên gia là một feed-forward network
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(model_dim, model_dim * 2),  # Mở rộng thành 2x
                nn.ReLU(),                             # Activation function
                nn.Linear(model_dim * 2, model_dim)   # Thu nhỏ về kích thước gốc
            ) 
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tiến propagation qua lớp MoE.
        
        Args:
            x (torch.Tensor): Input features có shape [batch_size, model_dim]
            
        Returns:
            torch.Tensor: Output sau khi đi qua MoE, cùng shape với input
        """
        # Lấy trọng số kết hợp và chỉ số top-k từ gating network
        combined_weights, top_k_indices, _ = self.gating(x)
        
        # Khởi tạo output cuối cùng
        final_output = torch.zeros_like(x)
        
        # Lặp qua từng chuyên gia
        for expert_idx in range(self.num_experts):
            # Tìm các mẫu được định tuyến đến chuyên gia này
            expert_mask = (top_k_indices == expert_idx)           # [batch_size, top_k]
            sample_mask = expert_mask.any(dim=1)                  # [batch_size] - các mẫu nào được định tuyến
            
            # Nếu không có mẫu nào được định tuyến, bỏ qua chuyên gia này
            if sample_mask.sum() == 0:
                continue
            
            # Lấy các feature tương ứng với các mẫu được chọn
            selected_features = x[sample_mask]
            
            # Xử lý qua chuyên gia
            expert_output = self.experts[expert_idx](selected_features)
            
            # Tính trọng số gating cho chuyên gia
            expert_weights = (combined_weights * expert_mask).sum(dim=1)
            selected_weights = expert_weights[sample_mask]
            
            # Cộng đóng góp của chuyên gia vào output cuối cùng (nhân với trọng số)
            final_output[sample_mask] += expert_output * selected_weights.unsqueeze(-1)
        
        return final_output

class MoEModel(nn.Module):
    """
    Mô hình Mixture of Experts cho phân loại ảnh bệnh lá cây.
    
    Cấu trúc:
    1. Feature Extractor: MobileNetV3 Large để trích xuất features từ ảnh
    2. MoE Layer: Định tuyến và xử lý features qua các chuyên gia
    3. Batch Normalization: Chuẩn hóa dữ liệu
    4. Classifier: Lớp fully-connected để phân loại
    
    Args:
        num_classes (int): Số lớp để phân loại
        num_experts (int): Số lượng chuyên gia trong MoE layer
        top_k (int): Số lượng chuyên gia hàng đầu được sử dụng
    """
    
    def __init__(self, num_classes: int, num_experts: int, top_k: int):
        super().__init__()
        
        # Trích xuất features từ ảnh input bằng MobileNetV3 Large
        self.feature_extractor = Mobilenetv3LargeFeatureExtractor()
        
        # Lớp MoE để xử lý features với nhiều chuyên gia
        self.moe_layer = MoELayer(
            model_dim=self.feature_extractor.output_dim,
            num_experts=num_experts,
            top_k=top_k
        )
        
        # Chuẩn hóa batch để ổn định quá trình huấn luyện
        self.normalizer = nn.BatchNorm1d(self.feature_extractor.output_dim)
        
        # Lớp phân loại: ánh xạ features vào num_classes
        self.classifier = nn.Linear(self.feature_extractor.output_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tiến propagation qua toàn bộ mô hình.
        
        Args:
            x (torch.Tensor): Ảnh input có shape [batch_size, 3, 224, 224]
            
        Returns:
            torch.Tensor: Logits có shape [batch_size, num_classes]
        """
        # Bước 1: Trích xuất features từ ảnh
        x = self.feature_extractor(x)
        
        # Bước 2: Chuẩn hóa batch
        x = self.normalizer(x)
        
        # Bước 3: Xử lý qua lớp MoE
        x = self.moe_layer(x)
        
        # Bước 4: Chuẩn hóa batch lần 2
        x = self.normalizer(x)
        
        # Bước 5: Phân loại
        x = self.classifier(x)
        
        return x


# ============================================================================
# KIỂM TRA VÀ TEST MÔ HÌNH
# ============================================================================
if __name__ == "__main__":
    # Cấu hình MoE Layer
    model_dim = 960          # Kích thước features từ MobileNetV3 Large
    num_experts = 4          # Số lượng chuyên gia
    top_k = 3                # Chọn top-3 chuyên gia tốt nhất
    
    print("=" * 60)
    print("TEST 1: Mixture of Experts Layer")
    print("=" * 60)
    
    # Tạo dummy input và kiểm tra MoE Layer
    dummy_input = torch.randn(3, model_dim)
    moe_layer = MoELayer(model_dim=model_dim, num_experts=num_experts, top_k=top_k)
    output = moe_layer(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # Kỳ vọng: [3, 960]
    print("✓ MoE Layer test passed!\n")
    
    print("=" * 60)
    print("TEST 2: Full MoE Model")
    print("=" * 60)
    
    # Tạo mô hình và kiểm tra với ảnh giả lập
    model = MoEModel(num_classes=8, num_experts=num_experts, top_k=top_k)
    dummy_image = torch.randn(32, 3, 224, 224)  # Batch 32 ảnh 224x224 RGB
    output = model(dummy_image)
    print(f"Input shape:  {dummy_image.shape}")
    print(f"Output shape: {output.shape}")  # Kỳ vọng: [32, 8]
    print("✓ MoE Model test passed!")
    print("=" * 60)