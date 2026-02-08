"""
Utility Functions and Dataset Loading Classes.

This module provides helper functions and dataset classes for loading and processing
plant disease images. It includes the LoadDataset class for structured data loading
with support for train/validation/test splits.

Key Features:
    - Automatic dataset splitting with stratification
    - Class to index mapping for label encoding
    - Image loading with PIL
    - Transform pipeline support via Albumentations
"""

import os
from typing import List, Tuple, Dict, Literal, Optional
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from PIL import Image
from sklearn.model_selection import train_test_split


class LoadDataset(Dataset):
    """
    Dataset loader for plant disease classification.
    
    Loads images from a structured directory of disease classes and provides
    automatic train/validation/test splitting with stratification to ensure
    balanced class distribution across splits.
    
    The dataset expects a directory structure like:
        root_dir/
        ├── Tomato_Bacterial_spot/
        ├── Tomato_Early_blight/
        ├── Tomato_healthy/
        └── ... (other disease classes)
    
    Attributes:
        root_dir (Path): Root directory containing disease class folders
        split (str): Dataset split ('train', 'val', or 'test')
        train_ratio (float): Proportion of data for training
        image_paths (List[str]): List of image file paths for the selected split
        labels (List[int]): Class labels corresponding to image_paths
        class_to_idx (Dict[str, int]): Mapping from class name to class index
        idx_to_class (Dict[int, str]): Mapping from class index to class name
    """

    def __init__(
        self,
        root_dir: Path,
        split: Literal['train', 'validation', 'test'],
        train_ratio: float = 0.8,
        transform: transforms.Compose = None
    ) -> None:
        """
        Initialize the dataset loader.
        
        Args:
            root_dir (Path): Root directory containing class subdirectories
            split (str, optional): Dataset split - 'train', 'val', or 'test'. Defaults to 'train'.
            train_ratio (float, optional): Proportion of data for training (0 to 1). Defaults to 0.8.
            transform (transforms.Compose, optional): Image transformation pipeline. Defaults to None.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.train_ratio = train_ratio
        self.image_paths, self.labels, self.class_to_idx, self.idx_to_class = self._split_dataset()

    def _load_image(self, root_dir: Path) -> Tuple[List[str], List[int], Dict[str, int], Dict[int, str]]:
        """
        Load all images and labels from the root directory.
        
        Scans the root directory for subdirectories starting with "Tomato" (disease classes),
        collects all image files from each class, and creates class-to-index mappings.
        
        Args:
            root_dir (Path): Root directory containing class subdirectories
            
        Returns:
            Tuple containing:
                - image_paths (List[str]): Absolute paths to all image files
                - labels (List[int]): Class labels (0-indexed) corresponding to each image
                - class_to_idx (Dict[str, int]): Mapping from class name to class index
                - idx_to_class (Dict[int, str]): Mapping from class index to class name
        """
        class_names = sorted(
            [d for d in os.listdir(root_dir)
             if os.path.isdir(os.path.join(root_dir, d))
             and d.startswith("Tomato")]
        )
        class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
        image_paths = []
        labels = []
        for class_name in class_names:
            dir = os.path.join(root_dir, class_name)
            for fname in os.listdir(dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(dir, fname))
                    labels.append(class_to_idx[class_name])
        return image_paths, labels, class_to_idx, idx_to_class

    def _split_dataset(self) -> Tuple[List[str], List[int], Dict[str, int], Dict[int, str]]:
        """
        Split dataset into train, validation, and test sets.
        
        Uses stratified sampling to ensure balanced class distribution across all splits:
        - 80% training, 20% temporary (from train_ratio)
        - Temporary split: 50% validation, 50% test
        
        Returns:
            Tuple containing:
                - image_paths (List[str]): Image paths for the selected split
                - labels (List[int]): Labels for the selected split
                - class_to_idx (Dict[str, int]): Class name to index mapping
                - idx_to_class (Dict[int, str]): Index to class name mapping
                
        Raises:
            ValueError: If split is not 'train', 'val', or 'test'
        """
        image_paths, labels, class_to_idx, idx_to_class = self._load_image(self.root_dir)

        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            image_paths, labels, test_size= 1-self.train_ratio, stratify=labels, random_state=42, shuffle=True
        )

        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42, shuffle=True
        )

        if self.split == 'train':
            return train_paths, train_labels, class_to_idx, idx_to_class
        elif self.split == 'validation':
            return val_paths, val_labels, class_to_idx, idx_to_class
        elif self.split == 'test':
            return test_paths, test_labels, class_to_idx, idx_to_class
        else:
            raise ValueError("split must be 'train', 'validation', or 'test'")

    def __len__(self) -> int:
        """
        Return the total number of samples in this dataset split.
        
        Returns:
            int: Number of images in the current split
        """
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample from the dataset by index.
        
        Loads an image from disk, applies transformations if specified, and returns
        the transformed image tensor and its corresponding class label.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            Tuple containing:
                - image (torch.Tensor): Transformed image tensor
                - label (int): Class label (0-indexed)
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        if self.transform:
            augumented = self.transform(image=image)
            image = augumented["image"]
        return image, label
