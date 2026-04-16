from utils.utils import LoadDataset
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import Counter

cropped_data_path = Path(__file__).resolve().parents[2] / 'data' / 'tomato_only'

# Augmentation cho tập huấn luyện
train_transform = A.Compose([
    # Geometric — giúp model không phụ thuộc vị trí lá trong ảnh
    A.Resize(256, 256),              # resize lên 256 trước
    A.RandomCrop(224, 224),          # crop ngẫu nhiên → tăng diversity hơn Resize(224) thẳng
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),           # lá cây lật dọc vẫn hợp lý
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.1,
        rotate_limit=30,
        p=0.5
    ),
 
    # Color — quan trọng nhất cho plant disease vì bệnh thay đổi màu lá
    # A.RandomBrightnessContrast(
    #     brightness_limit=0.2,
    #     contrast_limit=0.2,
    #     p=0.4
    # ),
    # A.HueSaturationValue(
    #     hue_shift_limit=10,
    #     sat_shift_limit=25,
    #     val_shift_limit=15,
    #     p=0.35
    # ),
    # # CLAHE: tăng tương phản cục bộ → làm rõ viền đốm bệnh
    # A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.3),
 
    # # Noise nhẹ — tránh overfit với ảnh quá sạch
    # A.OneOf([
    #     A.GaussNoise(var_limit=(5.0, 30.0), p=1.0),
    #     A.GaussianBlur(blur_limit=(3, 5), p=1.0),
    # ], p=0.2),
 
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2()
])

# Augmentation cho tập validation
val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2()
])

# Augmentation cho tập test 
test_transform = A.Compose([
    A.Resize(height=224, width=224),  # Resize giống transforms.Resize(224)
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()  # Thay cho transforms.ToTensor()
])

train_dataset = LoadDataset(
    root_dir=cropped_data_path,
    split='train',
    train_ratio=0.8,
    transform=train_transform
)

validation_dataset = LoadDataset(
    root_dir=cropped_data_path,
    split='validation',
    train_ratio=0.8,
    transform=val_transform
)

test_dataset = LoadDataset(
    root_dir=cropped_data_path,
    split='test',
    train_ratio=0.8,
    transform=test_transform
)

print(f"Train size: {len(train_dataset)}")
print(f"Validation size: {len(validation_dataset)}")
print(f"Test size: {len(test_dataset)}")
print(f"Numbers of train labels: {Counter(train_dataset.labels)}")
print(f"Numbers of validation labels: {Counter(validation_dataset.labels)}")
print(f"Numbers of test labels: {Counter(test_dataset.labels)}")
print(train_dataset.class_to_idx)

