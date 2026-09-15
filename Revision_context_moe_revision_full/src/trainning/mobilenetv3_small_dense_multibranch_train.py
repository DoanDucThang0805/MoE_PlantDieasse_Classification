"""
Train MobileNetV3-Small dense multi-branch model.

Examples:
    python src/trainning/mobilenetv3_small_dense_multibranch_train.py --seed 42
    python src/trainning/mobilenetv3_small_dense_multibranch_train.py --seeds 42 43 44
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from dataset.plantdoc_dataset import build_datasets
from models.dense_multibranch.mobilenetv3_small_dense_multibranch import (
    MobileNetV3SmallDenseMultiBranch,
)
from utils.dense_multibranch_trainner import DenseMultiBranchTrainer


warnings.filterwarnings("ignore")


BATCH_SIZE = 64
NUM_EPOCHS = 200
NUM_EXPERTS = 4
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = ArgumentParser(
        description="Training MobileNetV3-Small Dense Multi-Branch",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--seed", type=int, default=42, help="Single random seed")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="List of random seeds. When set, this overrides --seed.",
    )
    parser.add_argument(
        "--num_experts",
        type=int,
        default=NUM_EXPERTS,
        help="Number of dense branches/experts",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=NUM_EPOCHS,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=WEIGHT_DECAY,
        help="Weight decay",
    )
    parser.add_argument(
        "--type_model",
        type=str,
        default="dense_multibranch",
        help="Model type used in checkpoint path",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm. Use 0 to disable clipping.",
    )

    return parser.parse_args()


def train_one_seed(args, seed: int) -> None:
    set_seed(seed)

    print("\n===== Training Configuration =====")
    for key, value in vars(args).items():
        print(f"{key:<15}: {value}")
    print(f"{'current_seed':<15}: {seed}")
    print("==================================\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path.cwd().parents[0]

    checkpoint_dir = (
        output_dir
        / "checkpoints"
        / "plantdoc"
        / args.type_model
        / "mobilenetv3small_dense_multibranch"
        / f"{args.num_experts}_experts"
        / f"seed_{seed}"
    )

    train_dataset, validation_dataset, _ = build_datasets(use_context=False)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    labels = train_dataset.labels
    classes = np.unique(labels)
    num_classes = len(classes)

    print(f"Using device: {device}")
    print(f"Number of classes: {num_classes}")
    print(f"Checkpoint dir: {checkpoint_dir}")

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    )
    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device,
    )

    model = MobileNetV3SmallDenseMultiBranch(
        num_classes=num_classes,
        num_experts=args.num_experts,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    max_grad_norm = None if args.max_grad_norm <= 0 else args.max_grad_norm

    trainer = DenseMultiBranchTrainer(
        num_epochs=args.num_epochs,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        batch_size=args.batch_size,
        checkpoints_dir=str(checkpoint_dir),
        max_grad_norm=max_grad_norm,
    )

    trainer.train()


def main() -> None:
    args = get_args()
    seeds = args.seeds if args.seeds is not None else [args.seed]

    for seed in seeds:
        train_one_seed(args, seed)


if __name__ == "__main__":
    main()
