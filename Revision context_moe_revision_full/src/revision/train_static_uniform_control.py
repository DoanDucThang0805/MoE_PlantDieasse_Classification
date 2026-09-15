from __future__ import annotations
import argparse, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

from models.controls.static_uniform_expert import StaticUniformExpertModel
from utils.trainer import Trainer
from revision.common import get_dataset_builder


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset', default='plantdoc', choices=['plantdoc','slif'])
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--num_experts', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-3)
    ap.add_argument('--output_root', type=Path, default=Path('../checkpoints'))
    args=ap.parse_args(); set_seed(args.seed)

    builder, canonical = get_dataset_builder(args.dataset)
    train_ds, val_ds, _ = builder(use_context=False)
    g=torch.Generator(); g.manual_seed(args.seed)
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,generator=g)
    val_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False)
    classes=np.unique(train_ds.labels)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    w=compute_class_weight(class_weight='balanced', classes=classes, y=train_ds.labels)
    criterion=nn.CrossEntropyLoss(weight=torch.tensor(w,dtype=torch.float32,device=device))
    model=StaticUniformExpertModel(num_classes=len(classes),num_experts=args.num_experts)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer=optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out=args.output_root/canonical/'revision_controls'/'static_uniform_4expert'/f'seed_{args.seed}'
    trainer=Trainer(num_epochs=args.epochs, device=device, train_loader=train_loader,
                    val_loader=val_loader, model=model, criterion=criterion,
                    optimizer=optimizer, batch_size=args.batch_size, checkpoints_dir=str(out))
    trainer.train()

if __name__=='__main__': main()
