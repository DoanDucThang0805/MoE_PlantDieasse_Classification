from pathlib import Path
import random
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from dataset.plantdoc_dataset import build_datasets
from models.pretrained_model.ghostnet import model
from utils.trainer import Trainer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


parse = ArgumentParser()
parse.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
args = parse.parse_args()
set_seed(args.seed)

train_dataset, validation_dataset, _ = build_datasets(use_context=False)

BATCH_SIZE = 64
generator = torch.Generator()
generator.manual_seed(args.seed)

train_ds = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=generator,
)

val_ds = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
output_dir = Path.cwd().parents[0]

labels = train_dataset.labels
num_classes = len(set(labels))

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=labels,
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.001,
)

trainer = Trainer(
    num_epochs=200,
    device=device,
    batch_size=BATCH_SIZE,
    train_loader=train_ds,
    val_loader=val_ds,
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    checkpoints_dir=str(
        output_dir
        / "checkpoints"
        / "plantdoc"
        / "pretrain_models"
        / "ghostnet"
        / f"seed_{args.seed}"
    ),
)

if __name__ == "__main__":
    trainer.train()
