"""Ablation for the undocumented 6-D context path.

Architecture and parameter count are kept identical to the main Linear-Gating
MoE, but the six context features are replaced by zeros for every sample during
training/validation. This isolates whether image-derived context information,
rather than sparse routing itself, materially affects the reported gain.
"""
from __future__ import annotations
import argparse, random
from pathlib import Path
import numpy as np, torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.utils.class_weight import compute_class_weight
from models.moe.linear_model import MoEModel
from loss.loss_fn import MoELoss
from utils.moe_trainer import MoETrainer
from revision.common import get_dataset_builder

class ZeroContext(Dataset):
    def __init__(self, base): self.base=base; self.labels=base.labels
    def __len__(self): return len(self.base)
    def __getitem__(self,i):
        x,y,c=self.base[i]
        return x,y,torch.zeros_like(c)

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',default='plantdoc',choices=['plantdoc','slif']); ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--epochs',type=int,default=300); ap.add_argument('--batch_size',type=int,default=32); ap.add_argument('--output_root',type=Path,default=Path('../checkpoints'))
    args=ap.parse_args(); seed_all(args.seed); builder,canonical=get_dataset_builder(args.dataset); tr,va,_=builder(use_context=True); tr0,va0=ZeroContext(tr),ZeroContext(va)
    g=torch.Generator().manual_seed(args.seed); tl=DataLoader(tr0,batch_size=args.batch_size,shuffle=True,generator=g); vl=DataLoader(va0,batch_size=args.batch_size,shuffle=False)
    classes=np.unique(tr.labels); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); cw=compute_class_weight(class_weight='balanced',classes=classes,y=tr.labels); cw=torch.tensor(cw,dtype=torch.float32,device=dev)
    model=MoEModel(context_dim=6,num_classes=len(classes),num_experts=4,top_k=2,router_mode='context_aware',temperature=0.5); opt=optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3); loss=MoELoss(alpha=0.05,class_weights=cw)
    out=args.output_root/canonical/'revision_controls'/'moe_zero_context'/'4_experts'/'top_2'/f'seed_{args.seed}'
    MoETrainer(args.epochs,dev,tl,vl,model,loss,opt,args.batch_size,checkpoint_dir=out).train()

if __name__=='__main__': main()
