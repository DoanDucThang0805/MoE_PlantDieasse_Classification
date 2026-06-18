from utils.load_dataset import LoadDataset
from utils.context_features import extract_context_features
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import Counter


# Path to cropped tomato disease dataset
cropped_data_path = Path(__file__).resolve().parents[2] / 'data' / 'slif_tomato_dataset' / 'Phase_I-Dataset'
print("Data dir: ", cropped_data_path)

# Augmentation cho tập validation
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2()
])



def build_datasets(use_context: bool = True):
    """
    Build train, validation, and test datasets with optional context features.
    
    Creates three dataset instances (train, validation, test) using the LoadDataset class
    with predefined augmentation pipelines. Optionally extracts context features for each sample.

    Args:
        use_context (bool, optional): Whether to extract and include context features.
                                     Defaults to True.

    Returns:
        Tuple[LoadDataset, LoadDataset, LoadDataset]: A tuple containing the train, 
                                                      validation, and test datasets.
    """

    # Initialize context extractor if needed
    context_extractor = extract_context_features if use_context else None
    
    # Build training dataset with augmentation
    train_dataset = LoadDataset(
        root_dir=cropped_data_path,
        split="train",
        train_ratio=0.8,
        transform=transform,
        return_context=use_context,
        context_extractor=context_extractor,
    )

    # Build validation dataset with light augmentation (resize + normalize only)
    validation_dataset = LoadDataset(
        root_dir=cropped_data_path,
        split="validation",
        train_ratio=0.8,
        transform=transform,
        return_context=use_context,
        context_extractor=context_extractor,
    )

    # Build test dataset with light augmentation (resize + normalize only)
    test_dataset = LoadDataset(
        root_dir=cropped_data_path,
        split="test",
        train_ratio=0.8,
        transform=transform,
        return_context=use_context,
        context_extractor=context_extractor,
    )
    
    return train_dataset, validation_dataset, test_dataset


if __name__ == "__main__":
    """Test and validate dataset builder functionality."""
    
    print("==== Testing Dataset Builder ====\n")

    # Test WITHOUT context features
    print("Test 1: Dataset without context features")
    print("-" * 45)
    train_dataset, val_dataset, test_dataset = build_datasets(use_context=False)

    print(f"Train dataset size:       {len(train_dataset)}")
    print(f"Validation dataset size:  {len(val_dataset)}")
    print(f"Test dataset size:        {len(test_dataset)}")

    print("\nTrain label distribution:")
    print(Counter(train_dataset.labels))

    print("\nValidation label distribution:")
    print(Counter(val_dataset.labels))

    print("\nTest label distribution:")
    print(Counter(test_dataset.labels))

    print("\nClass mapping:")
    print(train_dataset.class_to_idx)

    # Retrieve and inspect a sample
    img, label, context = train_dataset[0]
    print("\nSample inspection (without context):")
    print(f"  Image shape:  {img.shape}")
    print(f"  Label:        {label}")
    print(f"  Context:      {context}")

    # Test WITH context features
    print("\n" + "="*45)
    print("Test 2: Dataset with context features")
    print("-" * 45)
    train_dataset_ctx, _, _ = build_datasets(use_context=True)

    img, label, context = train_dataset_ctx[0]

    print("\nSample inspection (with context):")
    print(f"  Image shape:    {img.shape}")

    if context is not None:
        print(f"  Context shape:  {context.shape}")
        print(f"  Context sample: {context[:5]}...")  # Show first 5 values
    else:
        print("  Context:        None (Error - context extraction failed)")

    print("\n==== Dataset Test Completed Successfully ====")
