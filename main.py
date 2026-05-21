from sklearn.model_selection import train_test_split

# Dữ liệu giả
image_paths = [f"img_{i}.jpg" for i in range(10)]
labels = [0,0,0,0,0,1,1,1,1,1]

train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.3,
    stratify=labels,
    random_state=42,
    shuffle=True
)

print("TRAIN:")
print(train_paths)

print("\nTEST:")
print(test_paths)

from torchvision import models
import torch
# Load 2 
model_pretrained = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
model_random     = models.mobilenet_v3_small(weights=None)
model_old_api    = models.mobilenet_v3_small(pretrained=True)

# So sánh weights layer đầu
w_pre = model_pretrained.features[0][0].weight.data
w_rnd = model_random.features[0][0].weight.data
w_old = model_old_api.features[0][0].weight.data

print(torch.allclose(w_old, w_pre))  # True  → pretrained ✅
print(torch.allclose(w_old, w_rnd))  # False → random     ❌