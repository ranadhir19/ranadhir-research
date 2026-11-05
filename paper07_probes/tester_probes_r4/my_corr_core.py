#!/usr/bin/env python3
"""R4 tester's OWN corrected/uncorrected adjusted computation.
Written fresh from the model description (standardised variables, priors
N(0,5^2) slopes / N(0,10^2) intercept, half-normal(1) sigma, analytic
coefficient integration + trapezoid quadrature over log sigma).
No builder code imported. Grid 601 by default (R3 showed grid error
<=2.3e-10 in log10BF at 601); field probe uses 1201.
Correction: Gram diagonal of specified covariate columns multiplied by
their lambda (per-column map); X'y, y'y unchanged."""
import numpy as np
from scipy.stats import norm

def _grid(n_grid):
    s = np.linspace(np.log(0.02), np.log(8.0), n_grid)
    sig = np.exp(s); ds = s[1]-s[0]
    w = np.full(n_grid, ds); w[0]*=0.5; w[-1]*=0.5
    return s, sig, w

def _logml(G, Xty, yty, n, pvar, focal, n_grid):
    s, sig, tw = _grid(n_grid)
    s2 = sig**2; p = G.shape[0]
    A = G[None]/s2[:,None,None] + np.diag(1.0/pvar)[None]
    try:
        L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        return None
    logdet = 2*np.log(np.diagonal(L,axis1=1,axis2=2)).sum(axis=1)
    rhs = Xty[None]/s2[:,None]
    m = np.linalg.solve(A, rhs[:,:,None])[:,:,0]
    logml_s = (-0.5*n*np.log(2*np.pi) - n*np.log(sig)
               -0.5*np.log(pvar).sum() -0.5*logdet
               -0.5*(yty/s2 - np.einsum('gi,gi->g', m, rhs)))
    li = logml_s + 0.5*np.log(2/np.pi) - 0.5*s2 + np.log(sig)
    mx = li.max(); wu = tw*np.exp(li-mx); logml = mx+np.log(wu.sum())
    w = wu/wu.sum()
    if focal is None: return logml, None
    ef = np.zeros((len(sig),p,1)); ef[:,focal,0]=1.0
    Ainv = np.linalg.solve(A, ef)[:,:,0]
    return logml, (m[:,focal], Ainv[:,focal], w)

def fit(y, x, covs, direction, lam_map=None, n_grid=601):
    """lam_map: dict covariate-index -> lambda for Gram correction.
    None/empty = uncorrected. Returns dict or {'evaluable':False}."""
    y=np.asarray(y,float); x=np.asarray(x,float)
    cols=[np.asarray(c,float) for c in covs]
    def z(v):
        sd=v.std(); return (v-v.mean())/sd if sd>0 else v-v.mean()
    zy,zx = z(y),z(x); zc=[z(c) for c in cols]; n=len(zy)
    def gram(X, cov_start, lam_map):
        G = X.T@X
        if lam_map:
            for ci,lam in lam_map.items():
                j = cov_start+ci; G[j,j]*=lam
        return G
    X1=np.column_stack([np.ones(n),zx]+zc)
    pv1=np.array([100.]+[25.]*(1+len(zc)))
    r1=_logml(gram(X1,2,lam_map), X1.T@zy, float(zy@zy), n, pv1, 1, n_grid)
    X0=np.column_stack([np.ones(n)]+zc)
    pv0=np.array([100.]+[25.]*len(zc))
    r0=_logml(gram(X0,1,lam_map), X0.T@zy, float(zy@zy), n, pv0, None, n_grid)
    if r1 is None or r0 is None: return {"evaluable":False}
    logml1,foc=r1; logml0,_=r0
    mf,vf,w = foc
    if direction>0: pd_=float(np.sum(w*norm.cdf(mf/np.sqrt(vf))))
    else: pd_=float(np.sum(w*norm.cdf(-mf/np.sqrt(vf))))
    logbf=float(logml1-logml0)
    return {"evaluable":True,"logBF":logbf,"log10BF":logbf/np.log(10),
            "Pd":pd_,"beta_mean":float(np.sum(w*mf)),"n":n}

def decide(f):
    if not f.get("evaluable"): return "not_evaluable"
    lb,pd_ = f["logBF"], f["Pd"]
    if lb>=np.log(10) and pd_>=0.95: return "accept"
    if np.log(3)<=lb<np.log(10) and pd_>=0.90: return "accept"
    if lb<=np.log(1/3): return "reject"
    return "inconclusive"

def lambda_min(dec_by_lam, grid=(1.0,0.9,0.8,0.7,0.6)):
    if dec_by_lam.get(1.0)!="accept": return None
    lm=1.0
    for lam in grid:
        if dec_by_lam.get(lam)=="accept": lm=lam
        else: break
    return lm
