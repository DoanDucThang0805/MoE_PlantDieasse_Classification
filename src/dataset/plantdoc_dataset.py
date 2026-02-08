from utils.utils import LoadDataset
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import Counter

cropped_data_path = Path(__file__).resolve().parents[2] / 'data' / 'tomato_only'

# Augmentation cho tập huấn luyện
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.05,
        rotate_limit=30,
        p=0.5
    ),
    # A.RandomGamma(p=0.2),
    # A.RandomBrightnessContrast(p=0.3),
    # A.RGBShift(
    #     r_shift_limit=15,
    #     g_shift_limit=15,
    #     b_shift_limit=15,
    #     p=0.3
    # ),
    # A.CLAHE(
    #     clip_limit=4.0,
    #     p=0.3
    # ),
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

