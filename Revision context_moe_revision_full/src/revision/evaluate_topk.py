from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import accuracy_score, f1_score

from models.moe.linear_model import MoEModel
from revision.common import SEEDS, make_loader, find_checkpoint


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',default='plantdoc',choices=['plantdoc','slif'])
    ap.add_argument('--checkpoint_root',type=Path,default=Path('../checkpoints'))
    ap.add_argument('--batch_size',type=int,default=32)
    ap.add_argument('--output_dir',type=Path,default=Path('../diagnostics/revision/topk'))
    args=ap.parse_args(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader,ds,canonical=make_loader(args.dataset,'test',args.batch_size,True)
    rows=[]
    for k in (1,2,3,4):
        root=args.checkpoint_root/canonical/'revision_topk_linear'/'mobilenetv3small_moe'/'4_experts'/f'top_{k}'
        for seed in SEEDS:
            try: ckpt=find_checkpoint(root,seed)
            except FileNotFoundError as e:
                print('[MISSING]',e); continue
            state=torch.load(ckpt,map_location=device)
            model=MoEModel(context_dim=state['context_dim'],num_classes=state['num_classes'],num_experts=4,
                           top_k=k,router_mode=state['router_mode'],temperature=state['temperature'])
            model.load_state_dict(state['model_state_dict']); model.to(device).eval()
            yp=[]; yt=[]
            with torch.inference_mode():
                for x,y,c in loader:
                    logits,_,_=model(x.to(device),c.to(device)); yp.append(logits.argmax(1).cpu().numpy()); yt.append(y.numpy())
            yp=np.concatenate(yp); yt=np.concatenate(yt)
            rows.append({'top_k':k,'seed':seed,'accuracy':accuracy_score(yt,yp),
                         'macro_f1':f1_score(yt,yp,average='macro',zero_division=0),'checkpoint':str(ckpt)})
            print(k,seed,rows[-1]['accuracy'],rows[-1]['macro_f1'])
    args.output_dir.mkdir(parents=True,exist_ok=True)
    raw=pd.DataFrame(rows); raw.to_csv(args.output_dir/'topk_seed_metrics.csv',index=False)
    if len(raw):
        summary=(raw.groupby('top_k').agg(n_seeds=('seed','count'),accuracy_mean=('accuracy','mean'),accuracy_std=('accuracy','std'),
                 macro_f1_mean=('macro_f1','mean'),macro_f1_std=('macro_f1','std')).reset_index())
        summary.to_csv(args.output_dir/'topk_summary.csv',index=False); print(summary.to_string(index=False))

if __name__=='__main__': main()
