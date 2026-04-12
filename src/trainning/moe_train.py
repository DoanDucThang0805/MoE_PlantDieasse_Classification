"""
Script Huấn Luyện Mô Hình MoE
============================
Huấn luyện mô hình Mixture of Experts (MoE) cho phân loại bệnh thực vật.
"""

from pathlib import Path
import numpy as np
import random
import warnings
import argparse

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight

from utils.moe_trainer import MoETrainer
from dataset.plantdoc_dataset import train_dataset, validation_dataset
from models.moe.model import MoEModel
from loss.loss_fn import MoELoss

warnings.filterwarnings("ignore")


# =============================================================================
# Seed Utility
# =============================================================================

def set_seed(seed: int):
    """Set seed cho reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Default Hyperparameters
# =============================================================================

BATCH_SIZE = 32
SHUFFLE_TRAIN = True
SHUFFLE_VAL = False

NUM_EXPERTS = 2
TOP_K = 2

NUM_EPOCHS = 300
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001
MOE_LOSS_ALPHA = 0.05


# =============================================================================
# Argument Parser
# =============================================================================

def get_args():
    parser = argparse.ArgumentParser(
        description="Training MoE Model for Plant Disease Classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument(
        "--type_model",
        type=str,
        default="MoE",
        help="Model type"
    )

    parser.add_argument(
        "--num_experts",
        type=int,
        default=NUM_EXPERTS,
        help="Number of experts"
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=TOP_K,
        help="Top-k experts selected"
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=NUM_EPOCHS,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help="Learning rate"
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=WEIGHT_DECAY,
        help="Weight decay"
    )

    parser.add_argument(
        "--moe_alpha",
        type=float,
        default=MOE_LOSS_ALPHA,
        help="MoE auxiliary loss weight"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for gating softmax"
    )

    return parser.parse_args()


# =============================================================================
# Main Training
# =============================================================================

def main():

    args = get_args()
    print("\n===== Training Configuration =====")
    for k, v in vars(args).items():
        print(f"{k:<15}: {v}")
    print("==================================\n")

    # -------------------------------------------------------------------------
    # Seed
    # -------------------------------------------------------------------------

    set_seed(args.seed)

    print(f"\nUsing seed: {args.seed}")

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------------------
    # Output Directory
    # -------------------------------------------------------------------------

    output_dir = Path.cwd().parents[0]

    checkpoint_dir = (
        output_dir
        / "checkpoints"
        / "plantdoc"
        / args.type_model
        / "mobilenetv3small_moe"
        / f"{args.num_experts}_experts"
        / f"top_{args.top_k}"
        / f"seed_{args.seed}"
    )

    # -------------------------------------------------------------------------
    # DataLoader (seeded)
    # -------------------------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=SHUFFLE_TRAIN,
    )

    val_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=SHUFFLE_VAL
    )

    # -------------------------------------------------------------------------
    # Dataset Info
    # -------------------------------------------------------------------------

    labels = train_dataset.labels
    num_classes = len(set(labels))

    print(f"Number of classes: {num_classes}")

    # -------------------------------------------------------------------------
    # Class Weights
    # -------------------------------------------------------------------------

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=labels
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32
    ).to(device)

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    model = MoEModel(
        num_classes=num_classes,
        num_experts=args.num_experts,
        top_k=args.top_k,
        temperature=args.temperature,
    )

    # -------------------------------------------------------------------------
    # Loss
    # -------------------------------------------------------------------------

    criterion = MoELoss(alpha=args.moe_alpha)

    # -------------------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------------------

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # -------------------------------------------------------------------------
    # Trainer
    # -------------------------------------------------------------------------

    trainer = MoETrainer(
        num_epochs=args.num_epochs,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        batch_size=args.batch_size,
        checkpoint_dir=checkpoint_dir
    )

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------

    trainer.train()


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    main()