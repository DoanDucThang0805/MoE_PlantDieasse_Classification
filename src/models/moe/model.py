import torch
import torch.nn as nn
from torchinfo import summary
from typing import Tuple
from .backbone import Mobilenetv3LargeFeatureExtractor
from .gating import NoisyTopKGating
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
    
    def __init__(self, model_dim: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.num_experts: int = num_experts
        self.top_k: int = top_k
        
        # Gating network: quyết định chuyên gia nào xử lý từng mẫu
        self.gating: NoisyTopKGating = NoisyTopKGating(
            model_dim=model_dim, 
            num_experts=num_experts, 
            top_k=top_k
        )
        
        # Danh sách các chuyên gia - mỗi chuyên gia là một feed-forward network
        self.experts: nn.ModuleList = nn.ModuleList([
            nn.Sequential(
                nn.Linear(model_dim, model_dim * 2),  # Mở rộng thành 2x
                nn.ReLU(),                             # Activation function
                nn.Linear(model_dim * 2, model_dim)   # Thu nhỏ về kích thước gốc
            ) 
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tiến propagation qua lớp MoE với routing dựa trên Gating Network.
        
        Args:
            x (torch.Tensor): Input features có shape [batch_size, model_dim]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (moe_output, clean_logits, top_k_indices)
                - moe_output: [batch_size, model_dim] - output từ experts
                - clean_logits: [batch_size, num_experts] - gating logits
                - top_k_indices: [batch_size, top_k] - top-k expert indices
        """
        # Bước 1: Lấy routing decisions từ Gating Network
        combined_weights: torch.Tensor
        top_k_indices: torch.Tensor
        clean_logits: torch.Tensor
        combined_weights, top_k_indices, clean_logits = self.gating(x)
        
        # Bước 2: Khởi tạo output tensor - sẽ cộng dồn đóng góp từ các experts
        moe_output: torch.Tensor = torch.zeros_like(x)
        
        # Bước 3: Routing & Expert Forward - gửi data tới từng expert
        for expert_idx in range(self.num_experts):
            # Tạo mask để tìm mẫu được gửi tới expert hiện tại
            expert_mask: torch.Tensor = (top_k_indices == expert_idx)      # [batch_size, top_k]
            sample_mask: torch.Tensor = expert_mask.any(dim=1)             # [batch_size] - mẫu nào được chọn
            
            # Tối ưu: skip nếu không có mẫu nào được gửi tới expert
            if sample_mask.sum() == 0:
                continue
            
            # Lấy features của mẫu được gửi tới expert này
            selected_features: torch.Tensor = x[sample_mask]
            
            # Xử lý qua expert feed-forward network
            expert_output: torch.Tensor = self.experts[expert_idx](selected_features)
            
            # Tính weighted sum - trọng số của expert này từ gating network
            expert_weights: torch.Tensor = (combined_weights * expert_mask).sum(dim=1)
            selected_weights: torch.Tensor = expert_weights[sample_mask]
            
            # Cộng đóng góp của expert vào output cuối cùng (weighted combination)
            moe_output[sample_mask] += expert_output * selected_weights.unsqueeze(-1)
        
        return moe_output, clean_logits, top_k_indices


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
    
    def __init__(self, num_classes: int, num_experts: int, top_k: int) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.num_experts = num_experts
        self.top_k = top_k

        
        # Trích xuất features từ ảnh input bằng MobileNetV3 Large
        self.feature_extractor: Mobilenetv3LargeFeatureExtractor = Mobilenetv3LargeFeatureExtractor()
        
        # Lớp MoE để xử lý features với nhiều chuyên gia
        self.moe_layer: MoELayer = MoELayer(
            model_dim=self.feature_extractor.output_dim,
            num_experts=num_experts,
            top_k=top_k
        )
        
        # Chuẩn hóa batch để ổn định quá trình huấn luyện
        self.normalizer: nn.BatchNorm1d = nn.BatchNorm1d(self.feature_extractor.output_dim)
        
        # Lớp phân loại: ánh xạ features vào num_classes
        self.classifier: nn.Linear = nn.Linear(self.feature_extractor.output_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tiến propagation qua toàn bộ mô hình Mixture of Experts.
        
        Pipeline: Image → Feature Extraction → BatchNorm → MoE → BatchNorm → Classifier
        
        Args:
            x (torch.Tensor): Ảnh input [batch_size, 3, 224, 224]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (logits, clean_logits, top_k_indices)
                - logits: [batch_size, num_classes] - classification output
                - clean_logits: [batch_size, num_experts] - gating routing info
                - top_k_indices: [batch_size, top_k] - which experts used
        """
        # Bước 1: Feature Extraction - trích xuất semantic features từ ảnh
        # Input: [batch_size, 3, 224, 224] → Output: [batch_size, 960]
        x: torch.Tensor = self.feature_extractor(x)
        
        # Bước 2: BatchNorm - chuẩn hóa để ổn định MoE layer
        x = self.normalizer(x)
        
        # Bước 3: Mixture of Experts - routing features qua multiple experts
        # Return: features + routing metadata cho analysis
        moe_output, clean_logits, top_k_indices = self.moe_layer(x)
        
        # Bước 4: BatchNorm - chuẩn hóa output sau MoE trước classifier
        x = self.normalizer(moe_output)
        
        # Bước 5: Classifier - ánh xạ features sang class predictions
        # Input: [batch_size, 960] → Output: [batch_size, num_classes]
        logits: torch.Tensor = self.classifier(x)
        
        return logits, clean_logits, top_k_indices


# ============================================================================
# KIỂM TRA VÀ TEST MÔ HÌNH
# ============================================================================
if __name__ == "__main__":
    # Cấu hình MoE Layer
    model_dim: int = 960          # Kích thước features từ MobileNetV3 Large
    num_experts: int = 4          # Số lượng chuyên gia
    top_k: int = 3                # Chọn top-3 chuyên gia tốt nhất
    
    print("=" * 60)
    print("TEST 1: Mixture of Experts Layer")
    print("=" * 60)
    
    # Tạo MoE Layer và test với dummy input
    dummy_input: torch.Tensor = torch.randn(3, model_dim)
    moe_layer: MoELayer = MoELayer(model_dim=model_dim, num_experts=num_experts, top_k=top_k)
    
    # Forward pass: trả về output, gating logits, và expert indices
    moe_output: torch.Tensor
    gating_logits: torch.Tensor
    expert_indices: torch.Tensor
    moe_output, gating_logits, expert_indices = moe_layer(dummy_input)
    
    print(f"Input shape:            {dummy_input.shape}")
    print(f"MoE Output shape:       {moe_output.shape}")      # [3, 960]
    print(f"Gating Logits shape:    {gating_logits.shape}")   # [3, 4]
    print(f"Top-k Indices shape:    {expert_indices.shape}")  # [3, 3]
    print("✓ MoE Layer test passed!\n")
    
    print("=" * 60)
    print("TEST 2: Full MoE Model")
    print("=" * 60)
    
    # Tạo full MoE Model và test với ảnh giả lập
    model: MoEModel = MoEModel(num_classes=8, num_experts=num_experts, top_k=top_k)
    dummy_image: torch.Tensor = torch.randn(32, 3, 224, 224)  # Batch 32 ảnh 224x224 RGB
    
    # Forward pass: trả về classification logits + routing metadata
    class_logits: torch.Tensor
    gating_logits: torch.Tensor
    expert_indices: torch.Tensor
    class_logits, gating_logits, expert_indices = model(dummy_image)
    
    print(f"Input shape:            {dummy_image.shape}")     # [32, 3, 224, 224]
    print(f"Class Logits shape:     {class_logits.shape}")    # [32, 8]
    print(f"Gating Logits shape:    {gating_logits.shape}")   # [32, 4]
    print(f"Top-k Indices shape:    {expert_indices.shape}")  # [32, 3]
    print("\n✓ Full MoE Model pipeline test passed!")
    print("=" * 60)
    summary(model, input_size=(1, 3, 224, 224), col_names=["input_size", "output_size", "num_params", "trainable"])
    print(model)
