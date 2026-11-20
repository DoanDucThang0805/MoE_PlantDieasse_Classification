"""Model Evaluation and Inference Script
======================================
This module performs inference on test datasets using trained models
and generates comprehensive evaluation reports including classification metrics, confusion matrices,
and performance visualizations.

Features:
    - Load trained model checkpoints
    - Batch inference on test data
    - Generate classification reports
    - Visualize confusion matrices
    - Create performance maps
"""

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from dataset.plantdoc_dataset import build_datasets
from models.pretrained_model.mobilenetv3_small import model


# ============================================================================
# Main Configuration
# ============================================================================

# Model and checkpoint information
MODEL_NAME = 'mobilenetv3_small'  # Model name (matches checkpoint directory name)
MODEL_TYPE = 'pretrain_models'
RUN_TIME = 'run_20260516-031725'  # Timestamp of training run
DATASET_NAME = 'plantdoc'
SEED = 46

# Data loading parameters
BATCH_SIZE = 32
SHUFFLE_TEST = True

# Plot image parameters
CONFUSION_MATRIX_FIGSIZE = (10,8)
CLASSIFICATION_REPORT_FIGSIZE = (10, 6)
REPORT_DPI = 300

# ============================================================================
# Device and Path Configuration
# ============================================================================

# Determine device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Determine checkpoint path (relative to this file location, handles different CWDs)
checkpoint_path = (
    Path(__file__).resolve().parents[3] / 'checkpoints' / DATASET_NAME / MODEL_TYPE / 
    MODEL_NAME / f'seed_{SEED}' / RUN_TIME / 'best_checkpoint.pth'
)

# Determine report directory
report_dir = (
    Path(__file__).resolve().parents[3] / 'reports' / DATASET_NAME / MODEL_TYPE / MODEL_NAME /  f'seed_{SEED}' / RUN_TIME
)

# ============================================================================
# Load Test Data
# ============================================================================

# _, _, test_dataset = build_datasets(False)
_, _, test_dataset = build_datasets(use_context=False)

# Create data loader for test set
test_ds = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE_TEST
)


# ============================================================================
# Load Checkpoint
# ============================================================================

# Check checkpoint existence
print(f"Loading checkpoint from: {checkpoint_path}")
if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

# Load checkpoint supporting multiple formats
checkpoint = torch.load(checkpoint_path, map_location=device)

# Support different key names in checkpoint
if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    # If checkpoint itself is state_dict, use it directly
    state_dict = checkpoint

# Load model to device and set evaluation mode
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# ============================================================================
# Inference on Test Set
# ============================================================================

# Lists to store all predictions and actual labels
all_preds = []
all_labels = []

# Perform inference on each batch
print("Performing inference on test set...")
with torch.inference_mode(True):
    for images, labels, _ in test_ds:
        # Move data to device
        images, labels = images.to(device), labels.to(device)
        
        # Perform inference (model returns logits, auxiliary loss and expert assignment)
        logits = model(images)
        
        # Compute probabilities and predict labels
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        # Store results
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

print(f"Inference completed. Total {len(all_labels)} samples processed.")

# ============================================================================
# Generate Classification Report
# ============================================================================
acc = accuracy_score(all_labels, all_preds)
print('Accuracy: ', acc)
# Get class names from dataset
target_names = [test_dataset.idx_to_class[i] for i in range(len(test_dataset.idx_to_class))]
print("\nDisease classes:")
print(target_names)

# Print detailed classification report
print("\n" + "="*80)
print("CLASSIFICATION REPORT")
print("="*80)
print(classification_report(all_labels, all_preds, target_names=target_names))

# ============================================================================
# Initialize Report Directory
# ============================================================================

# Create report directory if it doesn't exist
report_dir.mkdir(parents=True, exist_ok=True)
print(f"\nSaving reports to: {report_dir}")

# ============================================================================
# Visualize Confusion Matrix
# ============================================================================

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Create confusion matrix plot
fig, ax = plt.subplots(figsize=CONFUSION_MATRIX_FIGSIZE)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_names,
    yticklabels=target_names,
    annot_kws={"size": 15},
    linewidths=0.5,
    ax=ax,
)

ax.set_xlabel("Predicted Label", fontsize=15, labelpad=10)
ax.set_ylabel("True Label", fontsize=15, labelpad=10)
# ax.set_title("Confusion Matrix - Tomato Disease Classification", fontsize=14, pad=12)
ax.tick_params(axis='x', labelsize=15, rotation=45)
ax.tick_params(axis='y', labelsize=15, rotation=0)
ax.set_xticklabels(ax.get_xticklabels(), ha='right', fontsize=15)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)

# Colorbar font
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=15)

plt.tight_layout()

plt.savefig(report_dir / "confusion_matrix.png", dpi=REPORT_DPI, bbox_inches="tight")
print(f"✓ Saved confusion matrix: {report_dir / 'confusion_matrix.png'}")
plt.show()
# ============================================================================
# Visualize Classification Report
# ============================================================================

# Create classification report as dictionary
report_dict = classification_report(
    all_labels,
    all_preds,
    target_names=target_names,
    output_dict=True
)

# Convert to DataFrame for visualization
df = pd.DataFrame(report_dict).transpose()

# Create heatmap for classification report
plt.figure(figsize=CLASSIFICATION_REPORT_FIGSIZE)
sns.heatmap(
    df.iloc[:-1, :-1],
    annot=True,
    cmap="Blues",
    fmt=".2f",
    cbar_kws={"label": "Score"}
)

plt.title("Classification Report (Precision / Recall / F1-score)", fontsize=12)
plt.xlabel("Evaluation Metrics")
plt.ylabel("Disease Classes")
plt.tight_layout()

# Save image
plt.savefig(
    report_dir / "classification_report_heatmap.png",
    dpi=REPORT_DPI,
    bbox_inches="tight"
)
print(f"✓ Saved classification report: {report_dir / 'classification_report_heatmap.png'}")
plt.show()

print("\n" + "="*80)
print("MODEL EVALUATION COMPLETED")
print("="*80)
