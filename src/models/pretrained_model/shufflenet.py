from torchvision.models import shufflenet_v2_x2_0, ShuffleNet_V2_X2_0_Weights
from torchinfo import summary
import torch.nn as nn


num_classes = 8
weights = ShuffleNet_V2_X2_0_Weights.DEFAULT
model = shufflenet_v2_x2_0(weights=weights)
model.fc = nn.Linear(model.fc.in_features, num_classes)

summary(model, (1, 3, 224, 224), col_names=["input_size", "output_size", "num_params", "mult_adds"])
