"""Shuffled-routing counterfactual for Q8.

For each trained seed, extract the test-set features and learned Top-k routes.
Then permute complete (indices, weights) routing decisions across images while
keeping each image feature fixed. This preserves expert cardinality and the
empirical route distribution but destroys input-route correspondence.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import accuracy_score, f1_score

from models.moe.linear_model import MoEModel
from revision.common import SEEDS, make_loader, find_checkpoint

@torch.inference_mode()
def collect(model, loader, device):
    residuals=[]; norms=[]; idxs=[]; weights=[]; labels=[]
    for x,y,c in loader:
        x=x.to(device); c=c.to(device)
        feat=model.feature_extractor(x); zn=model.pre_moe_norm(feat)
        w,idx,_=model.moe_layer.gating(zn,c)
        residuals.append(feat.cpu()); norms.append(zn.cpu()); idxs.append(idx.cpu()); weights.append(w.cpu()); labels.append(y.cpu())
    return tuple(torch.cat(v,0) for v in (residuals,norms,idxs,weights,labels))

@torch.inference_mode()
def predict_with_routes(model,residual,zn,idx,w,device,batch=64):
    out=[]
    for s in range(0,len(zn),batch):
        r=residual[s:s+batch].to(device); z=zn[s:s+batch].to(device); ii=idx[s:s+batch].to(device); ww=w[s:s+batch].to(device)
        moe=torch.zeros_like(z)
        for e,expert in enumerate(model.moe_layer.experts):
            mask=(ii==e); sample=mask.any(dim=1)
            if sample.any():
                ew=(ww*mask).sum(dim=1)[sample]
                moe[sample]+=expert(z[sample])*ew.unsqueeze(-1)
        logits=model.classifier(model.post_moe_norm(r+moe)); out.append(logits.argmax(1).cpu())
    return torch.cat(out).numpy()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',default='plantdoc',choices=['plantdoc','slif'])
    ap.add_argument('--checkpoint_root',type=Path,default=Path('../checkpoints/plantdoc/moe_linearcontextaware_temp0.5/mobilenetv3small_moe/4_experts/top_2'))
    ap.add_argument('--permutations',type=int,default=100)
    ap.add_argument('--batch_size',type=int,default=32)
    ap.add_argument('--output_dir',type=Path,default=Path('../diagnostics/revision/routing_counterfactual'))
    args=ap.parse_args(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader,_,_=make_loader(args.dataset,'test',args.batch_size,True)
    rng=np.random.default_rng(20260915); rows=[]; seed_summary=[]
    for seed in SEEDS:
        ckpt=find_checkpoint(args.checkpoint_root,seed); state=torch.load(ckpt,map_location=device)
        model=MoEModel(context_dim=state['context_dim'],num_classes=state['num_classes'],num_experts=state['num_experts'],
                       top_k=state['top_k'],router_mode=state['router_mode'],temperature=state['temperature'])
        model.load_state_dict(state['model_state_dict']); model.to(device).eval()
        residual,zn,idx,w,y=collect(model,loader,device); y_np=y.numpy()
        pred=predict_with_routes(model,residual,zn,idx,w,device)
        learned_acc=accuracy_score(y_np,pred); learned_f1=f1_score(y_np,pred,average='macro',zero_division=0)
        p_acc=[]; p_f1=[]
        for p in range(args.permutations):
            perm=torch.as_tensor(rng.permutation(len(y)),dtype=torch.long)
            pp=predict_with_routes(model,residual,zn,idx[perm],w[perm],device)
            a=accuracy_score(y_np,pp); f=f1_score(y_np,pp,average='macro',zero_division=0); p_acc.append(a); p_f1.append(f)
            rows.append({'seed':seed,'permutation':p,'learned_accuracy':learned_acc,'learned_macro_f1':learned_f1,
                         'shuffled_accuracy':a,'shuffled_macro_f1':f})
        seed_summary.append({'seed':seed,'learned_accuracy':learned_acc,'learned_macro_f1':learned_f1,
                             'shuffled_accuracy_mean':np.mean(p_acc),'shuffled_accuracy_std':np.std(p_acc,ddof=1),
                             'shuffled_macro_f1_mean':np.mean(p_f1),'shuffled_macro_f1_std':np.std(p_f1,ddof=1),
                             'delta_accuracy':learned_acc-np.mean(p_acc),'delta_macro_f1':learned_f1-np.mean(p_f1)})
        print(seed,seed_summary[-1])
    args.output_dir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir/'shuffle_permutations.csv',index=False)
    sdf=pd.DataFrame(seed_summary); sdf.to_csv(args.output_dir/'shuffle_seed_summary.csv',index=False)
    stats=[]
    for m,learn,shuf in [('Accuracy','learned_accuracy','shuffled_accuracy_mean'),('Macro-F1','learned_macro_f1','shuffled_macro_f1_mean')]:
        a=sdf[learn].to_numpy(); b=sdf[shuf].to_numpy(); t=ttest_rel(a,b).pvalue
        try: wp=wilcoxon(a-b).pvalue
        except ValueError: wp=np.nan
        stats.append({'metric':m,'n_seeds':len(a),'learned_mean':a.mean(),'shuffled_mean':b.mean(),'mean_delta':(a-b).mean(),
                      'paired_t_p':t,'wilcoxon_p':wp})
    pd.DataFrame(stats).to_csv(args.output_dir/'shuffle_paired_statistics.csv',index=False)
    print(pd.DataFrame(stats).to_string(index=False))

if __name__=='__main__': main()
