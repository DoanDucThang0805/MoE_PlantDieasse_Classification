"""
Model Inference and Evaluation Script.

This module performs inference on a test dataset using a trained model and generates
comprehensive evaluation reports including classification metrics, confusion matrix,
and performance heatmaps.

Features:
    - Load trained model checkpoints
    - Batch inference on test data
    - Generate classification report
    - Create confusion matrix visualization
    - Generate performance heatmaps
"""

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from dataset.plantdoc_dataset import test_dataset
from models.moe import model


model_name = 'moe'
run_time = 'run_20260209-091118'
data = 'plantdoc'
num_class = 8
test_ds = DataLoader(test_dataset, batch_size=32, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Resolve checkpoint path relative to this file (robust across CWD)
checkpoint_path = Path(__file__).resolve().parents[2] / 'checkpoints' / data / "pretrained" / model_name / run_time / 'best_checkpoint.pth'
print(f"Loading checkpoint from: {checkpoint_path}")
if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


# Load checkpoint with support for different formats
checkpoint = torch.load(checkpoint_path, map_location=device)
# Support different checkpoint key names and provide a helpful error if missing
if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    # If the checkpoint itself is a state_dict, use it directly
    state_dict = checkpoint


# Load model to device
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# Run inference
all_preds = []
all_labels = []

with torch.inference_mode(True):
    for images, labels in test_ds:
        images, labels = images.to(device), labels.to(device)
        logits, _, _ = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# Generate classification report
target_names = [test_dataset.idx_to_class[i] for i in range(len(test_dataset.idx_to_class))]
print(target_names)
print(classification_report(all_labels, all_preds, target_names=target_names))


# Create output directory for reports (relative to workspace)
report_dir = Path(__file__).resolve().parents[2] / 'reports' / data / 'pretrain_weight' / model_name
report_dir.mkdir(parents=True, exist_ok=True)
print(f"Saving reports to: {report_dir}")

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Tomato Leaf Disease Classification")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(report_dir / "confusion_matrix.png")
plt.show()


# Tạo classification report dạng dict
report_dict = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True)
df = pd.DataFrame(report_dict).transpose()

plt.figure(figsize=(10, 6))
sns.heatmap(df.iloc[:-1, :-1], annot=True, cmap="Blues", fmt=".2f")
plt.title("Classification Report (Precision / Recall / F1-score)")
plt.savefig(report_dir / "classification_report_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()
