"""Controlled Linear-Gating MoE training for Q4: E=4, k=1..4.

Unlike the legacy trainning/moe_train.py, this script explicitly selects the
dataset, so PlantDoc and SLIF runs cannot be silently mixed.
"""
from __future__ import annotations
import argparse, random
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

from models.moe.linear_model import MoEModel
from loss.loss_fn import MoELoss
from utils.moe_trainer import MoETrainer
from revision.common import get_dataset_builder


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset', default='plantdoc', choices=['plantdoc','slif'])
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--num_experts', type=int, default=4)
    ap.add_argument('--top_k', type=int, required=True)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-3)
    ap.add_argument('--temperature', type=float, default=0.5)
    ap.add_argument('--moe_alpha', type=float, default=0.05)
    ap.add_argument('--context_dim', type=int, default=6)
    ap.add_argument('--output_root', type=Path, default=Path('../checkpoints'))
    args=ap.parse_args()
    if not 1 <= args.top_k <= args.num_experts: raise ValueError('Require 1 <= top_k <= num_experts')
    set_seed(args.seed)

    builder, canonical=get_dataset_builder(args.dataset)
    train_ds,val_ds,_=builder(use_context=True)
    g=torch.Generator(); g.manual_seed(args.seed)
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,generator=g)
    val_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False)
    classes=np.unique(train_ds.labels); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    w=compute_class_weight(class_weight='balanced',classes=classes,y=train_ds.labels)
    w=torch.tensor(w,dtype=torch.float32,device=device)
    model=MoEModel(context_dim=args.context_dim,num_classes=len(classes),num_experts=args.num_experts,
                   top_k=args.top_k,router_mode='context_aware',temperature=args.temperature)
    criterion=MoELoss(alpha=args.moe_alpha,class_weights=w)
    optimizer=optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    out=(args.output_root/canonical/'revision_topk_linear'/'mobilenetv3small_moe'/
         f'{args.num_experts}_experts'/f'top_{args.top_k}'/f'seed_{args.seed}')
    print(f'Checkpoint root: {out}')
    print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')
    trainer=MoETrainer(num_epochs=args.epochs,device=device,train_loader=train_loader,val_loader=val_loader,
                       model=model,criterion=criterion,optimizer=optimizer,batch_size=args.batch_size,
                       checkpoint_dir=out)
    trainer.train()

if __name__=='__main__': main()
