from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight

from .trainer import Trainer
from dataset.plantdoc_dataset import train_dataset, validation_dataset
from models.pretrained.mobilenetv4 import model


BATCH_SIZE = 64
train_ds = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_ds = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
output_dir = Path.cwd().parents[0]

labels = train_dataset.labels
num_classes = len(set(labels))

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(num_classes),
    y=labels
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

trainer = Trainer(
    num_epochs=200,
    device=device,
    batch_size=BATCH_SIZE,
    train_loader=train_ds,
    val_loader=val_ds,
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    checkpoints_dir=str(output_dir / "checkpoints" / "plantdoc" / "pretrain_weight" / "mobilenetv4")
)

if __name__ == "__main__":
    trainer.train()
