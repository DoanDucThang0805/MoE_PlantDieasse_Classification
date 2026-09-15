"""
Evaluation Metrics Module.

This module provides functions for computing evaluation metrics during model
training and inference, particularly for classification tasks.
"""

import torch

def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute the accuracy of predictions against true labels.
    
    Calculates the percentage of correct predictions by comparing predicted
    class indices with ground truth labels.

    Args:
        preds (torch.Tensor): Predicted class indices of shape (batch_size,)
        labels (torch.Tensor): True class indices of shape (batch_size,)

    Returns:
        float: Accuracy as a percentage (0-100)
        
    Example:
        >>> preds = torch.tensor([0, 1, 2, 0])
        >>> labels = torch.tensor([0, 1, 1, 0])
        >>> accuracy(preds, labels)
        75.0
    """
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return (correct / total) * 100.0
