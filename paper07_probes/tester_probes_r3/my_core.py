#!/usr/bin/env python3
"""Tester R3 OWN implementation. No builder code imported.
Marginal likelihood: coefficients integrated via Gaussian identity using
slogdet/solve on the posterior precision (mathematically forced form, coded
independently); sigma integrated by trapezoid on log-sigma.
Cross-validated in probe_r3_core.py against a brute-force n x n covariance
marginal (Woodbury/determinant-lemma in the DATA space, a different code path)
on small-n synthetic data.
"""
import numpy as np
from scipy.stats import norm

def make_grid(n_grid=1201, lo=0.02, hi=8.0):
    s = np.linspace(np.log(lo), np.log(hi), n_grid)
    return s, np.exp(s)

def log_ml_and_focal(y, X, prior_var, focal_idx, grid):
    s_grid, sig = grid
    n, p = X.shape
    s2 = sig**2
    XtX = X.T @ X; Xty = X.T @ y; yty = float(y @ y)
    prec_prior = 1.0/prior_var
    # per-sigma posterior precision
    A = XtX[None,:,:]/s2[:,None,None] + np.diag(prec_prior)[None,:,:]
    sign, logdet = np.linalg.slogdet(A)
    assert (sign>0).all()
    rhs = Xty[None,:]/s2[:,None]
    sol = np.linalg.solve(A, rhs[:,:,None])[:,:,0]  # posterior mean per sigma: A^{-1} X'y/sigma^2
    quad = yty/s2 - np.einsum('gi,gi->g', sol, rhs)
    log_ml_s = (-0.5*n*np.log(2*np.pi) - n*np.log(sig)
                -0.5*np.sum(np.log(prior_var)) -0.5*logdet -0.5*quad)
    # half-normal(1) prior on sigma, density in log-sigma variable: +log sigma jacobian
    log_int = log_ml_s + 0.5*np.log(2/np.pi) -0.5*s2 + np.log(sig)
    ds = s_grid[1]-s_grid[0]
    wts = np.full(len(sig), ds); wts[0]*=0.5; wts[-1]*=0.5
    mx = log_int.max()
    ew = wts*np.exp(log_int-mx)
    log_ml = mx + np.log(ew.sum())
    w = ew/ew.sum()
    if focal_idx is None:
        return log_ml, None
    # focal variance per sigma: diagonal of A^{-1}
    e = np.zeros((len(sig), p, 1)); e[:, focal_idx, 0]=1.0
    ainv_e = np.linalg.solve(A, e)[:,:,0]
    var_f = ainv_e[:, focal_idx]
    mean_f = sol[:, focal_idx]
    return log_ml, (mean_f, var_f, w)

def fit(y_raw, x_raw, cov_raw_list, direction, grid=None):
    if grid is None: grid = make_grid()
    def z(v):
        v=np.asarray(v,float); sd=v.std()  # ddof0
        return (v-v.mean())/sd if sd>0 else v-v.mean()
    zy, zx = z(y_raw), z(x_raw)
    zc = [z(c) for c in cov_raw_list]
    n=len(zy)
    X1=np.column_stack([np.ones(n), zx]+zc); pv1=np.array([100.]+[25.]*(1+len(zc)))
    X0=np.column_stack([np.ones(n)]+zc);     pv0=np.array([100.]+[25.]*len(zc))
    lm1, foc = log_ml_and_focal(zy, X1, pv1, 1, grid)
    lm0, _   = log_ml_and_focal(zy, X0, pv0, None, grid)
    mf, vf, w = foc
    if direction>0: pd_=float(np.sum(w*norm.cdf(mf/np.sqrt(vf))))
    else:           pd_=float(np.sum(w*norm.cdf(-mf/np.sqrt(vf))))
    logbf = lm1-lm0
    return {"log10BF": logbf/np.log(10), "logBF": float(logbf),
            "BF": float(np.exp(min(logbf,700))), "Pd": pd_,
            "beta_mean": float(np.sum(w*mf)), "n": int(n)}

def decide(bf, pd_):
    if bf>=10 and pd_>=0.95: return "accept"
    if 3<=bf<10 and pd_>=0.90: return "accept"
    if bf<=1/3: return "reject"
    return "inconclusive"

def p_claim_from_logbf(logbf, pd_, w):
    logit = np.log(w/(1-w)) + logbf
    pinc = 1/(1+np.exp(-logit)) if logit<700 else 1.0
    return pinc*pd_

def ledger_keep(pclaims, gamma=0.05):
    order=np.argsort([-p for p in pclaims]); keep=[]; cum=0.0
    for rank,i in enumerate(order):
        cum += 1-pclaims[i]
        if cum/(rank+1) <= gamma: keep.append(int(i))
        else: break
    return keep

def brute_log_ml(y, X, prior_var, sig):
    """Data-space marginal at FIXED sigma: y ~ N(0, sig^2 I + X diag(pv) X')."""
    n=len(y); C = sig**2*np.eye(n) + X @ np.diag(prior_var) @ X.T
    sign, ld = np.linalg.slogdet(C)
    return -0.5*n*np.log(2*np.pi) -0.5*ld -0.5*float(y @ np.linalg.solve(C, y))
