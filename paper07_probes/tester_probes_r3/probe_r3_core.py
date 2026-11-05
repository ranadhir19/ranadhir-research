#!/usr/bin/env python3
"""R3 core checks:
1. My log_ml formula vs brute-force data-space marginal at fixed sigmas (small n).
2. Empty-covariate certification recomputed with MY code on all 7 original edges
   (field samples rebuilt in probe_r3_field.py style -- here imported from that
   module's construction functions, which are the tester's own).
3. OVB identity (Prop 2) finite-sample OLS check on 3 field edges.
4. Prop 3 identity brute check + prefix-monotonicity check.
5. Grid-density sensitivity (401/801/1201/2001) on 3 edges.
Outputs probe_r3_core_results.json
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from my_core import fit, make_grid, log_ml_and_focal, brute_log_ml
from probe_r3_field import build_frames, edge_specs, complete_sample

out = {}

# --- 1. brute-force validation of the marginal formula at fixed sigma ---
rng = np.random.default_rng(42)
n=40
x=rng.standard_normal(n); c=rng.standard_normal(n); y=0.5*x+0.3*c+rng.standard_normal(n)
def z(v): return (v-v.mean())/v.std()
X=np.column_stack([np.ones(n), z(x), z(c)]); zy=z(y); pv=np.array([100.,25.,25.])
checks=[]
for sig in (0.5, 1.0, 1.7):
    # my per-sigma log ml: extract by calling the inner formula manually via a 1-pt trick:
    # recompute directly here (same algebra, but compared to data-space form below)
    s2=sig**2; A=X.T@X/s2+np.diag(1/pv); m=np.linalg.solve(A, X.T@zy/s2)
    ll_param = (-0.5*n*np.log(2*np.pi)-n*np.log(sig)-0.5*np.sum(np.log(pv))
                -0.5*np.linalg.slogdet(A)[1]-0.5*((zy@zy)/s2 - m@(X.T@zy/s2)))
    ll_data = brute_log_ml(zy, X, pv, sig)
    checks.append({"sigma":sig,"param_space":ll_param,"data_space":ll_data,"diff":abs(ll_param-ll_data)})
out["brute_force_fixed_sigma"]={"checks":checks,"max_diff":max(c["diff"] for c in checks)}

# --- 2+5. certification on the sample frames for the 7 original edges,
# empty covariates, my code ---
frames = build_frames()
specs = {s[0]:s for s in edge_specs(*frames)}
builder = json.loads((Path(__file__).parent.parent/"upgrade_probes"/"field_upgrade_results.json").read_text())
cert_builder = {e["id"]:e for e in builder["certification_empty_covariate_vs_existing_quadrature"]["edges"]}
cert=[]
grid_sens={}
for eid in list(cert_builder):
    _, fr, xc, yc, d, covs = specs[eid]
    dfr,_d = complete_sample(fr,[xc,yc]+covs)
    xv=dfr[xc].to_numpy(float); yv=dfr[yc].to_numpy(float)
    r = fit(yv, xv, [], d)
    cert.append({"id":eid,"my_log10BF_empty":r["log10BF"],
                 "builder_adjusted_empty":cert_builder[eid]["adjusted_empty_log10_BF10"],
                 "builder_existing_quad":cert_builder[eid]["existing_quad_log10_BF10"],
                 "diff_vs_builder_adj":abs(r["log10BF"]-cert_builder[eid]["adjusted_empty_log10_BF10"])})
    if eid in ("olist_delay_drives_review","telco_senior_drives_churn","bike_hum_drives_cnt"):
        pass
out["certification_my_code"]= {"edges":cert,"max_diff_vs_builder_adj":max(c["diff_vs_builder_adj"] for c in cert)}

# grid sensitivity on 3 edges (adjusted fits)
sens=[]
for eid in ("olist_freight_drives_review","telco_senior_drives_churn","bike_hum_drives_cnt"):
    _, fr, xc, yc, d, covs = specs[eid]
    dfr,_d = complete_sample(fr,[xc,yc]+covs)
    xv=dfr[xc].to_numpy(float); yv=dfr[yc].to_numpy(float); C=[dfr[c].to_numpy(float) for c in covs]
    row={"id":eid}
    for ng in (401,801,1201,2001):
        r=fit(yv,xv,C,d,grid=make_grid(ng))
        row[f"log10BF_grid{ng}"]=r["log10BF"]
    sens.append(row)
out["grid_sensitivity"]=sens

# --- 3. OVB identity, finite-sample OLS, 3 edges ---
ovb=[]
for eid in ("telco_senior_drives_churn","olist_freight_drives_review","bike_hum_drives_cnt"):
    _, fr, xc, yc, d, covs = specs[eid]
    dfr,_d = complete_sample(fr,[xc,yc]+covs)
    def z(v): v=np.asarray(v,float); return (v-v.mean())/v.std()
    yv=z(dfr[yc].to_numpy(float)); xv=z(dfr[xc].to_numpy(float))
    C=[z(dfr[c].to_numpy(float)) for c in covs]; nn=len(yv)
    Xadj=np.column_stack([np.ones(nn),xv]+C)
    coef,*_ = np.linalg.lstsq(Xadj,yv,rcond=None)
    beta_adj=coef[1]; gam=coef[2:]
    beta_biv=np.linalg.lstsq(np.column_stack([np.ones(nn),xv]),yv,rcond=None)[0][1]
    deltas=[]
    for cv in C:
        deltas.append(np.linalg.lstsq(np.column_stack([np.ones(nn),xv]),cv,rcond=None)[0][1])
    rhs = beta_adj + float(np.sum(np.array(gam)*np.array(deltas)))
    ovb.append({"id":eid,"beta_biv_ols":float(beta_biv),"beta_adj_ols":float(beta_adj),
                "gammas":[float(g) for g in gam],"deltas":[float(dd) for dd in deltas],
                "rhs_beta_adj_plus_sum":float(rhs),"identity_residual":float(abs(beta_biv-rhs))})
out["ovb_ols_identity"]=ovb

# --- 4. Prop 3 + prefix monotonicity, brute force on random claim probs ---
rng=np.random.default_rng(7); ok=True; worst=0.0
for trial in range(200):
    m=rng.integers(1,30); pc=rng.uniform(0,1,m)
    # posterior expected FDP for a FIXED set S = first k, by enumerating the identity directly
    S=list(range(m)); exp_fdp=np.mean([1-pc[i] for i in S])
    # simulation of posterior indicator expectation (Monte Carlo over claim truth)
    draws=np.array([rng.uniform(size=m)<pc for _ in range(4000)])
    mc=np.mean([np.mean(~draws[j][S]) for j in range(4000)])
    worst=max(worst, abs(exp_fdp-mc))
    # prefix means monotonicity for sorted descending pc
    q=np.sort(1-pc); means=np.cumsum(q)/np.arange(1,m+1)
    if np.any(np.diff(means)<-1e-12): ok=False
out["prop3_check"]={"mc_worst_abs_dev_vs_identity_over_200_random_sets":float(worst),
                    "prefix_means_monotone_all_trials":bool(ok),
                    "note":"MC tolerance ~0.01 at 4000 draws; identity itself is linearity, checked algebraically in report"}

Path(__file__).with_name("probe_r3_core_results.json").write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
