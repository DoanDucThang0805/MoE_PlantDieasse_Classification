from __future__ import annotations
import argparse, random
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from models.controls.matched_mobilenetv3_small import build_matched_baseline
from revision.common import get_dataset_builder
from utils.trainer import Trainer


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',default='plantdoc',choices=['plantdoc','slif']); ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--epochs',type=int,default=300); ap.add_argument('--batch_size',type=int,default=32); ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--weight_decay',type=float,default=1e-3)
    ap.add_argument('--output_root',type=Path,default=Path('../checkpoints')); args=ap.parse_args(); seed_all(args.seed)
    builder,canonical=get_dataset_builder(args.dataset); tr,va,_=builder(use_context=False); g=torch.Generator().manual_seed(args.seed)
    tl=DataLoader(tr,batch_size=args.batch_size,shuffle=True,generator=g); vl=DataLoader(va,batch_size=args.batch_size,shuffle=False)
    classes=np.unique(tr.labels); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); w=compute_class_weight(class_weight='balanced',classes=classes,y=tr.labels)
    model=build_matched_baseline(len(classes),pretrained=True); print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')
    criterion=nn.CrossEntropyLoss(weight=torch.tensor(w,dtype=torch.float32,device=device)); opt=optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    out=args.output_root/canonical/'revision_controls'/'matched_torchvision_mnv3small'/f'seed_{args.seed}'
    Trainer(args.epochs,device,tl,vl,model,criterion,opt,args.batch_size,checkpoints_dir=str(out)).train()

if __name__=='__main__': main()
