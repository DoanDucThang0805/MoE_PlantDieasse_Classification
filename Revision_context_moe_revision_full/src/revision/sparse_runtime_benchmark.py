"""Benchmark TRUE dynamic sparse PyTorch execution versus k on CPU.

Important: this is intentionally NOT the current ONNX export. The existing
model_adapt_onnx.py evaluates all experts for every sample, so its Raspberry Pi
numbers are a dense-export deployment measurement and cannot demonstrate Top-k
compute savings.
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from models.moe.linear_model import MoEModel


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',type=Path,required=True,help='A trained E=4 checkpoint; k can be overridden because parameter shapes do not depend on k.')
    ap.add_argument('--runs',type=int,default=300); ap.add_argument('--warmup',type=int,default=30)
    ap.add_argument('--threads',type=int,default=1); ap.add_argument('--output_csv',type=Path,default=Path('../diagnostics/revision/sparse_runtime_cpu.csv'))
    args=ap.parse_args(); torch.set_num_threads(args.threads); device=torch.device('cpu')
    state=torch.load(args.checkpoint,map_location='cpu'); sd=state['model_state_dict']; rows=[]
    torch.manual_seed(123); x=torch.randn(1,3,224,224); c=torch.randn(1,state.get('context_dim',6))
    for k in (1,2,3,4):
        model=MoEModel(context_dim=state.get('context_dim',6),num_classes=state.get('num_classes',8),num_experts=4,top_k=k,router_mode=state.get('router_mode','context_aware'),temperature=state.get('temperature',0.5))
        model.load_state_dict(sd); model.eval().to(device)
        with torch.inference_mode():
            for _ in range(args.warmup): model(x,c)
            ts=[]
            for _ in range(args.runs):
                t=time.perf_counter(); model(x,c); ts.append((time.perf_counter()-t)*1000)
        rows.append({'top_k':k,'runs':args.runs,'threads':args.threads,'mean_ms':np.mean(ts),'median_ms':np.median(ts),'p95_ms':np.percentile(ts,95),'std_ms':np.std(ts,ddof=1)})
        print(rows[-1])
    args.output_csv.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(args.output_csv,index=False)

if __name__=='__main__': main()
