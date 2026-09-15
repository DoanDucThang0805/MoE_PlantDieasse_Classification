"""Correct Q7 metrics across all five seeds.

Definitions:
  global_selection_share[e] = selections of expert e / (N*k), sum_e = 1.
  class_activation_rate[c,e] = images of class c selecting e / N_c, sum_e = k.
These are activation-frequency quantities, NOT mean routing weights.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from models.moe.linear_model import MoEModel
from revision.common import SEEDS, make_loader, find_checkpoint


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',default='plantdoc',choices=['plantdoc','slif'])
    ap.add_argument('--checkpoint_root',type=Path,default=Path('../checkpoints/plantdoc/moe_linearcontextaware_temp0.5/mobilenetv3small_moe/4_experts/top_2'))
    ap.add_argument('--output_dir',type=Path,default=Path('../diagnostics/revision/routing_usage'))
    ap.add_argument('--batch_size',type=int,default=32)
    args=ap.parse_args(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader,ds,_=make_loader(args.dataset,'test',args.batch_size,True)
    global_rows=[]; class_rows=[]
    for seed in SEEDS:
        ckpt=find_checkpoint(args.checkpoint_root,seed); state=torch.load(ckpt,map_location=device)
        model=MoEModel(context_dim=state['context_dim'],num_classes=state['num_classes'],num_experts=state['num_experts'],top_k=state['top_k'],router_mode=state['router_mode'],temperature=state['temperature'])
        model.load_state_dict(state['model_state_dict']); model.to(device).eval()
        counts=torch.zeros(model.num_experts); cc=torch.zeros(model.num_classes); ce=torch.zeros(model.num_classes,model.num_experts)
        with torch.inference_mode():
            for x,y,c in loader:
                _,_,idx=model(x.to(device),c.to(device)); idx=idx.cpu(); counts+=torch.bincount(idx.reshape(-1),minlength=model.num_experts)
                mask=F.one_hot(idx,num_classes=model.num_experts).sum(1).clamp(max=1).float()
                for cl in range(model.num_classes):
                    m=(y==cl); cc[cl]+=m.sum(); ce[cl]+=mask[m].sum(0)
        share=counts/counts.sum(); rates=ce/(cc[:,None]+1e-12)
        for e,v in enumerate(share): global_rows.append({'seed':seed,'expert':e,'global_selection_share':float(v)})
        for cl in range(model.num_classes):
            for e in range(model.num_experts): class_rows.append({'seed':seed,'class_id':cl,'class_name':ds.idx_to_class[cl], 'expert':e,'class_activation_rate':float(rates[cl,e])})
        print(f'seed {seed}: global sum={share.sum():.6f}; class row sums={rates.sum(1).numpy()}')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    g=pd.DataFrame(global_rows); c=pd.DataFrame(class_rows); g.to_csv(args.output_dir/'global_selection_share_per_seed.csv',index=False); c.to_csv(args.output_dir/'class_activation_rate_per_seed.csv',index=False)
    gs=g.groupby('expert').global_selection_share.agg(['mean','std']).reset_index(); gs.to_csv(args.output_dir/'global_selection_share_summary.csv',index=False)
    cs=c.groupby(['class_id','class_name','expert']).class_activation_rate.agg(['mean','std']).reset_index(); cs.to_csv(args.output_dir/'class_activation_rate_summary.csv',index=False)

if __name__=='__main__': main()
