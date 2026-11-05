#!/usr/bin/env python3
"""R4 severity probe: MY seeds, MY core (601 grid). Regimes:
 A  V3 replication, lambda_true=0.64 (noise sd 0.75), builder design
    (proxy only on confounded edges) -- procedures a,b,d(oracle),e(robust)
 B  lambda_true=0.50 (noise sd 1.0): b, d(oracle 0.5), e(grid 1..0.6),
    e_ext (grid extended to 0.5) -- tests grid floor
 C  lambda_true=0.80 (noise sd 0.5): b, d(oracle 0.8), e
 D  error correlated with x (rho(u, ex)=0.6, nominal lambda 0.64): b,d,e
 E  two covariates, only c1 proxied (all edge types, deployment design):
    b, d (oracle, scale ONLY proxied), e_only (robust, scale only proxied),
    e_all (robust, scale ALL covariates -- builder field generalisation)
 F  V2 spot-check (n=60, exact confounder): a,b only
15 worlds/regime (12 for F), 40 edges/world (16 true/12 null/12 conf)."""
import json, sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from my_corr_core import fit, decide, lambda_min
GRID=(1.0,0.9,0.8,0.7,0.6)

def summ(worlds, procs):
    out={}
    for p in procs:
        rat=np.array([w[p]["ratified"] for w in worlds],float)
        fal=np.array([w[p]["false"] for w in worlds],float)
        tru=np.array([w[p]["true"] for w in worlds],float)
        fdp=np.array([w[p]["fdp"] for w in worlds],float)
        out[p]={"ratified_mean":float(rat.mean()),"false_mean":float(fal.mean()),
                "true_mean":float(tru.mean()),"fdp_mean":float(fdp.mean())}
    return out

def run(seed, n_worlds, regime):
    rng=np.random.default_rng(seed); worlds=[]
    for wi in range(n_worlds):
        kinds=["true"]*16+["null"]*12+["conf"]*12
        order=rng.permutation(kinds); edges=[]
        for kind in order:
            kind=str(kind); n=60 if regime=="F" else 180
            if regime=="E":
                c1=rng.standard_normal(n); c2=rng.standard_normal(n)
                g1=rng.uniform(-0.4,0.4); g2=rng.uniform(-0.4,0.4)
                if kind=="true":
                    x=rng.standard_normal(n); beta=rng.uniform(0.25,0.60)
                    y=beta*x+g1*c1+g2*c2+rng.standard_normal(n)
                elif kind=="null":
                    x=rng.standard_normal(n); y=g1*c1+g2*c2+rng.standard_normal(n)
                else:
                    z=c1; lx,ly=rng.uniform(0.5,0.8),rng.uniform(0.5,0.8)
                    x=lx*z+np.sqrt(1-lx**2)*rng.standard_normal(n)
                    y=ly*z+g2*c2+np.sqrt(max(1-ly**2,0.05))*rng.standard_normal(n)
                # proxy of c1 for ALL edges, noise sd 0.75 (lambda 0.64)
                p1=c1+0.75*rng.standard_normal(n)
                covs=[p1,c2]
                s1=fit(y,x,[],1); d1=decide(s1)
                s2=fit(y,x,covs,1); d2=decide(s2)
                e={"kind":kind,"a":d1=="accept","b":d1=="accept" and d2=="accept"}
                # oracle scale-only-proxied at 0.64
                sd_=fit(y,x,covs,1,lam_map={0:0.64}); e["d"]=d1=="accept" and decide(sd_)=="accept"
                for name,lmfun in (("e_only",lambda lam:{0:lam}),("e_all",lambda lam:{0:lam,1:lam})):
                    dec={1.0:d2}
                    for lam in GRID[1:]:
                        dec[lam]=decide(fit(y,x,covs,1,lam_map=lmfun(lam)))
                    e[name]=d1=="accept" and lambda_min(dec)==0.6
                edges.append(e); continue
            # regimes A-D,F: single covariate
            if kind=="true":
                c=rng.standard_normal(n); x=rng.standard_normal(n)
                beta=rng.uniform(0.25,0.60); gamma=rng.uniform(-0.4,0.4)
                y=beta*x+gamma*c+rng.standard_normal(n); cov=c
            elif kind=="null":
                c=rng.standard_normal(n); x=rng.standard_normal(n)
                gamma=rng.uniform(-0.4,0.4); y=gamma*c+rng.standard_normal(n); cov=c
            else:
                z=rng.standard_normal(n); lx,ly=rng.uniform(0.5,0.8),rng.uniform(0.5,0.8)
                ex=rng.standard_normal(n)
                x=lx*z+np.sqrt(1-lx**2)*ex
                y=ly*z+np.sqrt(1-ly**2)*rng.standard_normal(n)
                if regime=="A": cov=z+0.75*rng.standard_normal(n)
                elif regime=="B": cov=z+1.0*rng.standard_normal(n)
                elif regime=="C": cov=z+0.5*rng.standard_normal(n)
                elif regime=="D":
                    eu=rng.standard_normal(n)
                    u=0.75*(0.6*ex+0.8*eu)  # Var(u)=0.5625, corr(u,ex)=0.6
                    cov=z+u
                else: cov=z
            s1=fit(y,x,[],1); d1=decide(s1)
            s2=fit(y,x,[cov],1); d2=decide(s2)
            e={"kind":kind,"a":d1=="accept","b":d1=="accept" and d2=="accept"}
            if regime in ("A","B","C","D"):
                lam_true={"A":0.64,"B":0.50,"C":0.80,"D":0.64}[regime]
                sd_=fit(y,x,[cov],1,lam_map={0:lam_true}); e["d"]=d1=="accept" and decide(sd_)=="accept"
                dec={1.0:d2}
                for lam in GRID[1:]:
                    dec[lam]=decide(fit(y,x,[cov],1,lam_map={0:lam}))
                e["e"]=d1=="accept" and lambda_min(dec)==0.6
                if regime=="B":
                    dec2=dict(dec); dec2[0.5]=decide(fit(y,x,[cov],1,lam_map={0:0.5}))
                    e["e_ext"]=d1=="accept" and dec2[0.5]=="accept" and lambda_min(dec)==0.6
            edges.append(e)
        procs=["a","b"]+(["d","e"] if regime in ("A","B","C","D") else [])+(["e_ext"] if regime=="B" else [])+(["d","e_only","e_all"] if regime=="E" else [])
        rec={}
        for p in procs:
            rat=[ee for ee in edges if ee[p]]; fal=[ee for ee in rat if ee["kind"]!="true"]
            rec[p]={"ratified":len(rat),"false":len(fal),"true":sum(ee["kind"]=="true" for ee in rat),
                    "fdp":len(fal)/len(rat) if rat else 0.0}
        worlds.append(rec)
        print(regime,"world",wi,{p:(rec[p]["ratified"],rec[p]["false"],rec[p]["true"]) for p in procs},flush=True)
    procs=list(worlds[0].keys())
    return {"regime":regime,"n_worlds":n_worlds,"summary":summ(worlds,procs),"worlds":worlds}

def main():
    out={}
    out["A_lambda0.64_replication"]=run(991001,15,"A")
    out["B_lambda0.50"]=run(991002,15,"B")
    out["C_lambda0.80"]=run(991003,15,"C")
    out["D_corr_error_x"]=run(991004,15,"D")
    out["E_two_cov_one_proxy"]=run(991005,15,"E")
    out["F_V2_n60_spotcheck"]=run(991006,12,"F")
    (HERE/"probe_r4_severity_results.json").write_text(json.dumps(out,indent=2))
    for k,v in out.items(): print(k, {p:(round(s["fdp_mean"],3),round(s["false_mean"],2),round(s["true_mean"],1)) for p,s in v["summary"].items()})

if __name__=="__main__": main()
