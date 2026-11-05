#!/usr/bin/env python3
"""R4 algebra probe: (1) reproduce Prop 4 in its stated case (single
covariate, standardised, classical independent error) with MY seed and
MY code; (2) scope probes the builder did not run: one proxy among two
covariates (scale-only-proxy vs scale-all), two proxies common vs
different lambda, error correlated with x, error correlated with the
outcome residual. Population covariances computed analytically from
structural equations. Output: probe_r4_algebra_results.json"""
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def xslope(S, idx_x=0, idx_cov=(1,), idx_y=None):
    """x-coefficient of population projection of y on (x, covs) from a
    covariance matrix (no intercept; means zero)."""
    cols=[idx_x]+list(idx_cov)
    A=S[np.ix_(cols,cols)]; b=S[cols,idx_y]
    return float(np.linalg.solve(A,b)[0])

def std_cov(S):
    d=np.sqrt(np.diag(S)); return S/np.outer(d,d)

def build(rng, scenario):
    """Structural model: z,c independent-ish latents; x,y linear.
    Returns dict of scenario population quantities."""
    # latents: z (true confounder), c (second covariate)
    rho_zc = rng.uniform(-0.4,0.4)
    a_xz, a_xc = rng.uniform(-0.8,0.8), rng.uniform(-0.6,0.6)
    beta = rng.uniform(-0.8,0.8)
    g_z, g_c = rng.uniform(-1,1), rng.uniform(-0.8,0.8)
    sd_e = rng.uniform(0.5,1.5)
    lam1 = rng.choice([0.5,0.64,0.8])
    lam2 = rng.choice([0.5,0.64,0.8])
    # latent covariance of (z,c): [[1,rho],[rho,1]]
    # x = a_xz z + a_xc c + ex, Var(ex)=1 (scale arbitrary; standardised later)
    # y = beta x + g_z z + g_c c + sd_e*e
    # components in basis (z,c,ex,e): 
    comp_x = np.array([a_xz, a_xc, 1.0, 0.0])
    comp_y = beta*comp_x + np.array([g_z, g_c, 0.0, sd_e])
    Sig_lat = np.eye(4); Sig_lat[0,1]=Sig_lat[1,0]=rho_zc
    # variables (x,z,c,y) covariance (unstandardised)
    V = np.array([comp_x, np.array([1,0,0,0]), np.array([0,1,0,0]), comp_y], float)
    S_true_raw = V @ Sig_lat @ V.T
    return dict(S_true_raw=S_true_raw, beta=beta, lam1=float(lam1), lam2=float(lam2),
                rho_zc=float(rho_zc))

def obs_cov(S_true_raw, scenario, rng, lam1, lam2):
    """Observed (x, z*, c(*), y) raw covariance under scenario; u vars
    appended as extra basis dims handled case by case via covariances:
    Cov(z*,.)=Cov(z,.)+Cov(u,.); Var(z*)=Var(z)+Var(u)+2Cov(z,u)."""
    S = S_true_raw.copy()  # order x,z,c,y (raw)
    varz = S[1,1]
    if scenario=="single":   # only z proxied; c exact but we will use only z as covariate
        varu = varz*(1-lam1)/lam1
        So = S.copy(); So[1,1]=varz+varu
        return So
    if scenario=="oneproxy_scaleonly" or scenario=="oneproxy_scaleall":
        varu = varz*(1-lam1)/lam1
        So = S.copy(); So[1,1]=varz+varu
        return So
    if scenario in ("twoproxy_common","twoproxy_diff"):
        varc=S[2,2]
        vu1 = varz*(1-lam1)/lam1
        l2 = lam1 if scenario=="twoproxy_common" else lam2
        vu2 = varc*(1-l2)/l2
        So=S.copy(); So[1,1]=varz+vu1; So[2,2]=varc+vu2
        return So
    if scenario=="corr_x":
        # u = k*ex_component + independent part, total Var(u) as lam1.
        # ex is basis dim 2 with coefficient in x = 1.0 (comp_x[2]=1)
        # Cov(u,x)=k*Var(ex)=k ; Cov(u,z)=Cov(u,c)=0; Cov(u,y)=k*beta (y contains beta*x incl beta*ex)
        varu = varz*(1-lam1)/lam1
        k = 0.6*np.sqrt(varu)  # correlation u-ex = 0.6
        So=S.copy(); So[1,1]=varz+varu
        So[0,1]=So[1,0]=S[0,1]+k
        So[1,3]=So[3,1]=S[1,3]+k*0.0  # u corr with ex only; y's ex coeff = beta*1 -> Cov(u,y)=k*beta
        So[1,3]=So[3,1]=S[1,3]+k*float(S[0,3]*0+ (np.nan if False else 0))  # placeholder overwritten below
        return So, k
    raise ValueError(scenario)

def main():
    rng=np.random.default_rng(424242)
    N=300
    res={}
    # --- Part 1: stated case, single covariate ---
    discs=[]
    for i in range(N):
        b=build(rng,"single"); Sraw=b["S_true_raw"]; lam=b["lam1"]
        Stru=std_cov(Sraw)
        true_slope=xslope(Stru,0,(1,),3)
        So=obs_cov(Sraw,"single",rng,lam,None)
        Sobs=std_cov(So)
        Sc=Sobs.copy(); Sc[1,1]=lam*Sobs[1,1]
        corr_slope=xslope(Sc,0,(1,),3)
        discs.append(abs(true_slope-corr_slope))
    res["stated_case_single_proxy_300_myseed"]={"max_abs_disc":float(max(discs)),"mean_abs_disc":float(np.mean(discs))}
    # --- Part 2a: two covariates, one proxy (z), c exact ---
    d_only=[]; d_all=[]
    for i in range(N):
        b=build(rng,"x"); Sraw=b["S_true_raw"]; lam=b["lam1"]
        Stru=std_cov(Sraw); true_slope=xslope(Stru,0,(1,2),3)
        So=obs_cov(Sraw,"oneproxy_scaleonly",rng,lam,None); Sobs=std_cov(So)
        Sc=Sobs.copy(); Sc[1,1]=lam*Sobs[1,1]
        d_only.append(abs(true_slope-xslope(Sc,0,(1,2),3)))
        Sa=Sobs.copy(); Sa[1,1]=lam*Sobs[1,1]; Sa[2,2]=lam*Sobs[2,2]
        d_all.append(abs(true_slope-xslope(Sa,0,(1,2),3)))
    res["two_cov_one_proxy"]={"scale_only_proxy_max":float(max(d_only)),"scale_only_proxy_mean":float(np.mean(d_only)),
                              "scale_ALL_max":float(max(d_all)),"scale_ALL_mean":float(np.mean(d_all))}
    # --- Part 2b: two proxies ---
    d_common=[]; d_diff_commonlam=[]; d_diff_ownlam=[]
    for i in range(N):
        b=build(rng,"x"); Sraw=b["S_true_raw"]; lam1,lam2=b["lam1"],b["lam2"]
        Stru=std_cov(Sraw); true_slope=xslope(Stru,0,(1,2),3)
        So=obs_cov(Sraw,"twoproxy_common",rng,lam1,lam2); Sobs=std_cov(So)
        Sc=Sobs.copy(); Sc[1,1]=lam1*Sobs[1,1]; Sc[2,2]=lam1*Sobs[2,2]
        d_common.append(abs(true_slope-xslope(Sc,0,(1,2),3)))
        So2=obs_cov(Sraw,"twoproxy_diff",rng,lam1,lam2); Sobs2=std_cov(So2)
        Sd=Sobs2.copy(); Sd[1,1]=lam1*Sobs2[1,1]; Sd[2,2]=lam1*Sobs2[2,2]
        d_diff_commonlam.append(abs(true_slope-xslope(Sd,0,(1,2),3)))
        Se=Sobs2.copy(); Se[1,1]=lam1*Sobs2[1,1]; Se[2,2]=lam2*Sobs2[2,2]
        d_diff_ownlam.append(abs(true_slope-xslope(Se,0,(1,2),3)))
    res["two_proxies"]={"common_lambda_scaleall_max":float(max(d_common)),
                        "diff_lambda_common_scaling_max":float(max(d_diff_commonlam)),"diff_lambda_common_scaling_mean":float(np.mean(d_diff_commonlam)),
                        "diff_lambda_own_scaling_max":float(max(d_diff_ownlam))}
    # --- Part 2c: error correlated with x (single covariate) ---
    d_corr=[]
    for i in range(N):
        b=build(rng,"x"); Sraw=b["S_true_raw"]; lam=b["lam1"]; beta=b["beta"]
        Stru=std_cov(Sraw); true_slope=xslope(Stru,0,(1,),3)
        varz=Sraw[1,1]; varu=varz*(1-lam)/lam; k=0.6*np.sqrt(varu)
        So=Sraw.copy(); So[1,1]=varz+varu
        So[0,1]=So[1,0]=Sraw[0,1]+k          # Cov(u,x)=k (ex coeff in x is 1, Var(ex)=1)
        So[1,3]=So[3,1]=Sraw[1,3]+k*beta     # Cov(u,y)=k * (ex coeff in y = beta)
        Sobs=std_cov(So); Sc=Sobs.copy(); Sc[1,1]=lam*Sobs[1,1]
        d_corr.append(abs(true_slope-xslope(Sc,0,(1,),3)))
    res["corr_error_with_x_rho0.6"]={"max_abs_disc":float(max(d_corr)),"mean_abs_disc":float(np.mean(d_corr)),
                                     "frac_settings_disc_gt_0.05":float(np.mean(np.array(d_corr)>0.05))}
    # --- Part 2d: error correlated with outcome residual e ---
    d_corr_e=[]
    for i in range(N):
        b=build(rng,"x"); Sraw=b["S_true_raw"]; lam=b["lam1"]
        Stru=std_cov(Sraw); true_slope=xslope(Stru,0,(1,),3)
        varz=Sraw[1,1]; varu=varz*(1-lam)/lam
        # y contains sd_e*e with e basis dim3; Cov(u,y)=0.6*sqrt(varu)*sd_e approx: recover sd_e from build? use Cov: y residual wrt (x,z,c) is sd_e*e; set Cov(u, e-part)=0.6*sqrt(varu)*gamma_e where gamma_e = coefficient: approximate via Sraw: Var of residual unknown here; instead directly add delta to Cov(z*,y) of size 0.3*sqrt(varu*Var(y)) (a generic differential-error perturbation)
        delta=0.3*np.sqrt(varu*Sraw[3,3])
        So=Sraw.copy(); So[1,1]=varz+varu; So[1,3]=So[3,1]=Sraw[1,3]+delta
        Sobs=std_cov(So); Sc=Sobs.copy(); Sc[1,1]=lam*Sobs[1,1]
        d_corr_e.append(abs(true_slope-xslope(Sc,0,(1,),3)))
    res["corr_error_with_y_perturbation"]={"max_abs_disc":float(max(d_corr_e)),"mean_abs_disc":float(np.mean(d_corr_e)),
                                           "frac_settings_disc_gt_0.05":float(np.mean(np.array(d_corr_e)>0.05))}
    (HERE/"probe_r4_algebra_results.json").write_text(json.dumps(res,indent=2))
    print(json.dumps(res,indent=2))

if __name__=="__main__": main()
