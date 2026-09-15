from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import accuracy_score, f1_score
from models.moe.linear_model import MoEModel
from models.controls.static_uniform_expert import StaticUniformExpertModel
from models.controls.matched_mobilenetv3_small import build_matched_baseline
from revision.common import SEEDS, make_loader, find_checkpoint


def load_state(path):
    x=torch.load(path,map_location='cpu'); return x.get('model_state_dict',x)

@torch.inference_mode()
def eval_model(model, loader, device, moe=False):
    yp=[]; yt=[]; model.to(device).eval()
    for x,y,c in loader:
        logits=model(x.to(device),c.to(device))[0] if moe else model(x.to(device))
        yp.append(logits.argmax(1).cpu().numpy()); yt.append(y.numpy())
    yp=np.concatenate(yp); yt=np.concatenate(yt)
    return accuracy_score(yt,yp), f1_score(yt,yp,average='macro',zero_division=0)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint_root',type=Path,default=Path('../checkpoints')); ap.add_argument('--output_dir',type=Path,default=Path('../diagnostics/revision/controls')); ap.add_argument('--batch_size',type=int,default=32); args=ap.parse_args()
    loader,_,_=make_loader('plantdoc','test',args.batch_size,True); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    roots={
      'MoE': args.checkpoint_root/'plantdoc'/'moe_linearcontextaware_temp0.5'/'mobilenetv3small_moe'/'4_experts'/'top_2',
      'Static-uniform 4-expert': args.checkpoint_root/'plantdoc'/'revision_controls'/'static_uniform_4expert',
      'Matched torchvision MNV3-Small': args.checkpoint_root/'plantdoc'/'revision_controls'/'matched_torchvision_mnv3small',
    }
    rows=[]
    for seed in SEEDS:
      for name,root in roots.items():
        try: ck=find_checkpoint(root,seed)
        except FileNotFoundError as e: print('[MISSING]',e); continue
        if name=='MoE':
          raw=torch.load(ck,map_location='cpu'); model=MoEModel(raw['context_dim'],raw['num_classes'],raw['num_experts'],raw['top_k'],raw['router_mode'],raw['temperature']); model.load_state_dict(raw['model_state_dict']); is_moe=True
        elif name.startswith('Static'):
          model=StaticUniformExpertModel(8,4,pretrained=False); model.load_state_dict(load_state(ck)); is_moe=False
        else:
          model=build_matched_baseline(8,pretrained=False); model.load_state_dict(load_state(ck)); is_moe=False
        a,f=eval_model(model,loader,dev,is_moe); rows.append({'seed':seed,'model':name,'accuracy':a,'macro_f1':f,'parameters':sum(p.numel() for p in model.parameters())}); print(seed,name,a,f)
    args.output_dir.mkdir(parents=True,exist_ok=True); df=pd.DataFrame(rows); df.to_csv(args.output_dir/'control_seed_metrics.csv',index=False)
    if df.empty: return
    summary=df.groupby('model').agg(n_seeds=('seed','count'),parameters=('parameters','first'),accuracy_mean=('accuracy','mean'),accuracy_std=('accuracy','std'),macro_f1_mean=('macro_f1','mean'),macro_f1_std=('macro_f1','std')).reset_index(); summary.to_csv(args.output_dir/'control_summary.csv',index=False)
    stats=[]
    for control in ['Static-uniform 4-expert','Matched torchvision MNV3-Small']:
      for metric in ['accuracy','macro_f1']:
        p=df.pivot(index='seed',columns='model',values=metric).dropna(subset=['MoE',control]);
        if len(p)<2: continue
        d=p['MoE'].to_numpy()-p[control].to_numpy(); pt=ttest_rel(p['MoE'],p[control]).pvalue
        try: pw=wilcoxon(d).pvalue
        except ValueError: pw=np.nan
        stats.append({'comparison':f'MoE vs {control}','metric':metric,'n_seeds':len(p),'mean_delta':d.mean(),'paired_t_p':pt,'wilcoxon_p':pw})
    pd.DataFrame(stats).to_csv(args.output_dir/'control_paired_statistics.csv',index=False); print(summary.to_string(index=False))

if __name__=='__main__': main()
