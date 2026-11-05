#!/usr/bin/env python3
"""R4 regime G: DIFFERENTIAL error -- proxy error u correlated with the
outcome's idiosyncratic component (confounded edges: y = ly*z +
sqrt(1-ly^2)*ey; u = 0.75*(0.6*ey + 0.8*eu), so Var(u)=0.5625, nominal
lambda 0.64, Cov(u,y)!=0, Cov(z*,y)!=Cov(z,y): Prop 4 premise violated
in the y direction). True/null edges as builder V3 (exact covariate).
Procedures a,b,d(oracle nominal 0.64),e. 12 worlds, seed 991007."""
import json, sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from my_corr_core import fit, decide, lambda_min
GRID=(1.0,0.9,0.8,0.7,0.6)
rng=np.random.default_rng(991007); worlds=[]
for wi in range(12):
    kinds=["true"]*16+["null"]*12+["conf"]*12; order=rng.permutation(kinds); edges=[]
    for kind in order:
        kind=str(kind); n=180
        if kind=="true":
            c=rng.standard_normal(n); x=rng.standard_normal(n)
            beta=rng.uniform(0.25,0.60); gamma=rng.uniform(-0.4,0.4)
            y=beta*x+gamma*c+rng.standard_normal(n); cov=c
        elif kind=="null":
            c=rng.standard_normal(n); x=rng.standard_normal(n)
            gamma=rng.uniform(-0.4,0.4); y=gamma*c+rng.standard_normal(n); cov=c
        else:
            z=rng.standard_normal(n); lx,ly=rng.uniform(0.5,0.8),rng.uniform(0.5,0.8)
            x=lx*z+np.sqrt(1-lx**2)*rng.standard_normal(n)
            ey=rng.standard_normal(n)
            y=ly*z+np.sqrt(1-ly**2)*ey
            eu=rng.standard_normal(n)
            u=0.75*(0.6*ey+0.8*eu)
            cov=z+u
        s1=fit(y,x,[],1); d1=decide(s1); s2=fit(y,x,[cov],1); d2=decide(s2)
        e={"kind":kind,"a":d1=="accept","b":d1=="accept" and d2=="accept"}
        sd_=fit(y,x,[cov],1,lam_map={0:0.64}); e["d"]=d1=="accept" and decide(sd_)=="accept"
        dec={1.0:d2}
        for lam in GRID[1:]: dec[lam]=decide(fit(y,x,[cov],1,lam_map={0:lam}))
        e["e"]=d1=="accept" and lambda_min(dec)==0.6
        edges.append(e)
    rec={}
    for p in ("a","b","d","e"):
        rat=[ee for ee in edges if ee[p]]; fal=[ee for ee in rat if ee["kind"]!="true"]
        rec[p]={"ratified":len(rat),"false":len(fal),"true":sum(ee["kind"]=="true" for ee in rat),
                "fdp":len(fal)/len(rat) if rat else 0.0}
    worlds.append(rec); print("G world",wi,{p:(rec[p]["ratified"],rec[p]["false"],rec[p]["true"]) for p in rec},flush=True)
summary={}
for p in ("a","b","d","e"):
    summary[p]={"fdp_mean":float(np.mean([w[p]["fdp"] for w in worlds])),
                "false_mean":float(np.mean([w[p]["false"] for w in worlds])),
                "true_mean":float(np.mean([w[p]["true"] for w in worlds]))}
out={"regime":"G_differential_error_u_corr_y_noise_rho0.6","n_worlds":12,"summary":summary,"worlds":worlds}
(HERE/"probe_r4_severity_G_results.json").write_text(json.dumps(out,indent=2))
print(json.dumps(summary,indent=2))
