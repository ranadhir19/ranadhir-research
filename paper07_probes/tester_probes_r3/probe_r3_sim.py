#!/usr/bin/env python3
"""R3 simulation falsification. Tester's own fit (my_core), own seeds.
Baseline: builder spec (16 true/12 null/12 confounded-only, n=180, confounder
supplied EXACTLY) with a NEW master seed, 10 worlds, full 1201 grid.
Variants (reduced 601 grid, validated against 1201 in core probe; 8 worlds
each unless noted): 
 V1 weak confounding loadings U(0.25,0.45)
 V2 n=60
 V3 noisy proxy: stage2 gets z + N(0,0.75^2) instead of z
 V4 unmeasured: stage2 gets an independent noise covariate instead of z
 V5 weak true effects beta U(0.10,0.30) (ledger-binding regime check)
 V6 true-edge covariate correlated with x (rho=0.7): power cost of stage 2
Each variant changes ONE thing from baseline. Outputs probe_r3_sim_results.json
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from my_core import fit, decide, make_grid, p_claim_from_logbf, ledger_keep

GRID_FAST = make_grid(601)

def run_worlds(seed, n_worlds, n_obs=180, conf_load=(0.50,0.80), beta_range=(0.25,0.60),
               proxy_noise=0.0, unmeasured=False, corr_true_cov=0.0, grid=None, label=""):
    rng=np.random.default_rng(seed)
    recs=[]
    for wi in range(n_worlds):
        kinds=["true"]*16+["null"]*12+["conf"]*12
        order=rng.permutation(kinds)
        edges=[]
        for kind in order:
            kind=str(kind)
            if kind=="true":
                c=rng.standard_normal(n_obs)
                if corr_true_cov>0:
                    x=corr_true_cov*c+np.sqrt(1-corr_true_cov**2)*rng.standard_normal(n_obs)
                else:
                    x=rng.standard_normal(n_obs)
                beta=rng.uniform(*beta_range); gamma=rng.uniform(-0.40,0.40)
                y=beta*x+gamma*c+rng.standard_normal(n_obs); cov=c
            elif kind=="null":
                c=rng.standard_normal(n_obs); x=rng.standard_normal(n_obs)
                gamma=rng.uniform(-0.40,0.40); y=gamma*c+rng.standard_normal(n_obs); cov=c
            else:
                z=rng.standard_normal(n_obs)
                lx,ly=rng.uniform(*conf_load),rng.uniform(*conf_load)
                x=lx*z+np.sqrt(1-lx**2)*rng.standard_normal(n_obs)
                y=ly*z+np.sqrt(1-ly**2)*rng.standard_normal(n_obs)
                if unmeasured: cov=rng.standard_normal(n_obs)
                elif proxy_noise>0: cov=z+proxy_noise*rng.standard_normal(n_obs)
                else: cov=z
            s1=fit(y,x,[],1,grid=grid); s2=fit(y,x,[cov],1,grid=grid)
            d1=decide(s1["BF"],s1["Pd"]); d2=decide(s2["BF"],s2["Pd"])
            pc=p_claim_from_logbf(s2["logBF"],s2["Pd"],0.5)
            edges.append({"kind":kind,"rat_a":d1=="accept","rat_b":d1=="accept" and d2=="accept","pc":pc})
        keep=ledger_keep([e["pc"] for e in edges if e["rat_b"]])
        rb=[i for i,e in enumerate(edges) if e["rat_b"]]
        for j,i in enumerate(rb): edges[i]["rat_c"]= j in keep
        for e in edges: e.setdefault("rat_c",False)
        rec={"world":wi}
        for proc,key in (("a","rat_a"),("b","rat_b"),("c","rat_c")):
            rat=[e for e in edges if e[key]]; false=[e for e in rat if e["kind"]!="true"]
            rec[proc]={"ratified":len(rat),"false":len(false),
                       "conf_ratified":sum(e["kind"]=="conf" for e in rat),
                       "true_ratified":sum(e["kind"]=="true" for e in rat),
                       "fdp":len(false)/len(rat) if rat else 0.0}
        rec["b_minus_c_ratified"]=rec["b"]["ratified"]-rec["c"]["ratified"]
        recs.append(rec)
        print(label, "world", wi, {p:(rec[p]["ratified"],rec[p]["false"]) for p in "abc"}, flush=True)
    def agg(proc,f): 
        v=np.array([r[proc][f] for r in recs],float); return {"mean":float(v.mean()),"min":float(v.min()),"max":float(v.max())}
    return {"label":label,"n_worlds":n_worlds,
            "summary":{p:{f:agg(p,f) for f in ("ratified","false","conf_ratified","true_ratified","fdp")} for p in "abc"},
            "worlds_with_c_differs_from_b":int(sum(r["b_minus_c_ratified"]>0 for r in recs)),
            "total_c_removed":int(sum(r["b_minus_c_ratified"] for r in recs)),
            "worlds":recs}

out={}
out["baseline_new_seed_777"]=run_worlds(777,10,grid=None,label="baseline")  # full 1201 grid
out["V1_weak_confounding"]=run_worlds(778,8,conf_load=(0.25,0.45),grid=GRID_FAST,label="V1")
out["V2_n60"]=run_worlds(779,8,n_obs=60,grid=GRID_FAST,label="V2")
out["V3_noisy_proxy_sd0.75"]=run_worlds(780,8,proxy_noise=0.75,grid=GRID_FAST,label="V3")
out["V4_unmeasured"]=run_worlds(781,8,unmeasured=True,grid=GRID_FAST,label="V4")
out["V5_weak_true_beta"]=run_worlds(782,8,beta_range=(0.10,0.30),grid=GRID_FAST,label="V5")
out["V6_corr_true_cov_rho0.7"]=run_worlds(783,8,corr_true_cov=0.7,grid=GRID_FAST,label="V6")
Path(__file__).with_name("probe_r3_sim_results.json").write_text(json.dumps(out,indent=2))
for k,v in out.items():
    print(k, {p:(round(v["summary"][p]["ratified"]["mean"],2),round(v["summary"][p]["false"]["mean"],2),round(v["summary"][p]["true_ratified"]["mean"],2),round(v["summary"][p]["fdp"]["mean"],3)) for p in "abc"}, "c_differs_worlds", v["worlds_with_c_differs_from_b"], "c_removed", v["total_c_removed"])
