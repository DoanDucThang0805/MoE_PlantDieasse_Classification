from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np,pandas as pd,torch
from scipy.stats import ttest_rel,wilcoxon
from sklearn.metrics import accuracy_score,f1_score
from models.moe.linear_model import MoEModel
from revision.common import SEEDS,make_loader,find_checkpoint

@torch.inference_mode()
def evaluate(model,loader,device,zero=False):
    yp=[];yt=[];model.to(device).eval()
    for x,y,c in loader:
        if zero: c=torch.zeros_like(c)
        logits,_,_=model(x.to(device),c.to(device)); yp.append(logits.argmax(1).cpu().numpy());yt.append(y.numpy())
    yp=np.concatenate(yp);yt=np.concatenate(yt);return accuracy_score(yt,yp),f1_score(yt,yp,average='macro',zero_division=0)

def load(path,device):
    s=torch.load(path,map_location=device);m=MoEModel(s['context_dim'],s['num_classes'],s['num_experts'],s['top_k'],s['router_mode'],s['temperature']);m.load_state_dict(s['model_state_dict']);return m

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint_root',type=Path,default=Path('../checkpoints'));ap.add_argument('--output_dir',type=Path,default=Path('../diagnostics/revision/context_ablation'));args=ap.parse_args();dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');loader,_,_=make_loader('plantdoc','test',32,True)
    roots={'Main context-aware':args.checkpoint_root/'plantdoc'/'moe_linearcontextaware_temp0.5'/'mobilenetv3small_moe'/'4_experts'/'top_2','Zero-context trained':args.checkpoint_root/'plantdoc'/'revision_controls'/'moe_zero_context'/'4_experts'/'top_2'};rows=[]
    for seed in SEEDS:
        for name,r in roots.items():
            try: ck=find_checkpoint(r,seed)
            except FileNotFoundError as e: print('[MISSING]',e);continue
            a,f=evaluate(load(ck,dev),loader,dev,zero=(name=='Zero-context trained'));rows.append({'seed':seed,'model':name,'accuracy':a,'macro_f1':f})
    args.output_dir.mkdir(parents=True,exist_ok=True);df=pd.DataFrame(rows);df.to_csv(args.output_dir/'context_seed_metrics.csv',index=False)
    stats=[]
    for metric in ['accuracy','macro_f1']:
        p=df.pivot(index='seed',columns='model',values=metric).dropna();
        if len(p)<2:continue
        a=p['Main context-aware'].to_numpy();b=p['Zero-context trained'].to_numpy();d=a-b
        try:pw=wilcoxon(d).pvalue
        except ValueError:pw=np.nan
        stats.append({'metric':metric,'n_seeds':len(a),'context_mean':a.mean(),'zero_context_mean':b.mean(),'mean_delta':d.mean(),'paired_t_p':ttest_rel(a,b).pvalue,'wilcoxon_p':pw})
    pd.DataFrame(stats).to_csv(args.output_dir/'context_paired_statistics.csv',index=False);print(pd.DataFrame(stats).to_string(index=False))
if __name__=='__main__':main()
