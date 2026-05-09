from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import torch
import torch.nn as nn
from torchinfo import summary


class Expert(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Hardswish(),
            nn.Dropout(0.2),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


class MobileNetV3SmallDenseMultiBranch(nn.Module):

    def __init__(
        self,
        num_classes=8,
        num_experts=4
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_experts = num_experts

        weights = MobileNet_V3_Small_Weights.DEFAULT
        backbone = mobilenet_v3_small(weights=weights)

        # backbone
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.flatten = nn.Flatten(1)

        # feature dimension
        in_features = backbone.classifier[0].in_features

        # multi-branch experts
        self.experts = nn.ModuleList([
            Expert(in_features, num_classes)
            for _ in range(num_experts)
        ])


    def forward(self, x):

        # backbone feature extraction
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)

        # run all experts
        expert_outputs = []

        for expert in self.experts:
            out = expert(x)
            expert_outputs.append(out)

        # [num_experts, batch_size, num_classes]
        expert_outputs = torch.stack(expert_outputs, dim=0)

        # average logits
        final_output = expert_outputs.mean(dim=0)

        return final_output


if __name__ == "__main__":
    model = MobileNetV3SmallDenseMultiBranch(
        num_classes=8,
        num_experts=4
    )

    summary(model, input_size=(1, 3, 224, 224))
