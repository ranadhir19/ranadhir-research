#!/usr/bin/env python3
"""Paper 7 upgrade: deterministic adjusted (multivariate) Bayesian computation.

Protocol frozen in ../upgrade_preregistration.md:
  standardised variables (z-scores, ddof=0); model y = X theta + eps with
  X = [1, focal x, covariates...]; priors theta_j ~ N(0, 5^2) independently,
  intercept N(0, 10^2); sigma ~ HalfNormal(scale = max(sd(z_y),1) = 1).
  Coefficients are integrated ANALYTICALLY conditional on sigma (Gaussian
  posterior via the p x p precision / Woodbury form); sigma is integrated by
  trapezoid quadrature over log sigma on a fixed grid. The directional
  probability is the sigma-mixture posterior probability for the focal slope
  (per-sigma normal CDFs, quadrature weights). No MCMC anywhere.
"""
import numpy as np
from scipy.stats import norm

LOG_LO, LOG_HI, N_GRID = float(np.log(0.02)), float(np.log(8.0)), 1201
S_GRID = np.linspace(LOG_LO, LOG_HI, N_GRID)
SIG_GRID = np.exp(S_GRID)
DS = S_GRID[1] - S_GRID[0]
TRAP_W = np.full(N_GRID, DS); TRAP_W[0] *= 0.5; TRAP_W[-1] *= 0.5
LN2PI = float(np.log(2 * np.pi))


def _log_marginal_focal(XtX, Xty, yty, n, prior_var, focal_idx):
    """Return (log marginal likelihood, focal posterior mean, focal var-mean
    components) mixing over sigma. focal_idx None -> null model (no focal)."""
    p = XtX.shape[0]
    lam = 1.0 / prior_var                      # (p,) prior precisions
    s2 = SIG_GRID ** 2                         # (G,)
    A = XtX[None, :, :] / s2[:, None, None] + np.diag(lam)[None, :, :]
    L = np.linalg.cholesky(A)                 # (G,p,p)
    logdetA = 2.0 * np.log(np.diagonal(L, axis1=1, axis2=2)).sum(axis=1)
    rhs = Xty[None, :] / s2[:, None]           # A m = X'y / sigma^2
    m = np.linalg.solve(A, rhs[:, :, None])[:, :, 0]          # (G,p)
    # log p(y | sigma): Gaussian marginal via posterior precision form
    log_ml_s = (-0.5 * n * LN2PI - n * np.log(SIG_GRID)
                - 0.5 * float(np.log(prior_var).sum())
                - 0.5 * logdetA
                - 0.5 * (yty / s2 - np.einsum('gi,gi->g', m, rhs)))
    # sigma prior: half-normal scale 1, and Jacobian d sigma = sigma d(log sigma)
    log_integrand = (log_ml_s + 0.5 * np.log(2.0 / np.pi)
                     - 0.5 * s2 + np.log(SIG_GRID))
    mx = float(log_integrand.max())
    w_unnorm = TRAP_W * np.exp(log_integrand - mx)
    log_ml = mx + float(np.log(w_unnorm.sum()))
    w = w_unnorm / w_unnorm.sum()              # posterior weights over sigma grid
    if focal_idx is None:
        return log_ml, None, None
    # focal posterior variance per sigma: (A^{-1})_ff via solving A v = e_f
    ef = np.zeros((len(SIG_GRID), p, 1)); ef[:, focal_idx, 0] = 1.0
    Ainv_ef = np.linalg.solve(A, ef)[:, :, 0]
    var_f = Ainv_ef[:, focal_idx]
    mean_f = float(np.sum(w * m[:, focal_idx]))
    return log_ml, (m[:, focal_idx], var_f, w), None


def fit_adjusted(y, x_focal, covs, direction):
    """y, x_focal: raw (n,) arrays; covs: list of raw (n,) arrays (may be []).
    Returns dict with log10 BF10 (adjusted pipeline), focal posterior mean
    (standardised scale), and directional probability Pd. With covs == [] this
    is the bivariate (stage-1) computation of the same protocol."""
    y = np.asarray(y, float); x = np.asarray(x_focal, float)
    cols = [np.asarray(c, float) for c in covs]
    def z(v):
        sd = v.std()  # ddof=0, per pre-registration
        return (v - v.mean()) / sd if sd > 0 else v - v.mean()
    zy, zx = z(y), z(x)
    zc = [z(c) for c in cols]
    n = len(zy)
    # alternative: [1, x, covs]; null: [1, covs] (focal fixed at 0)
    X1 = np.column_stack([np.ones(n), zx] + zc)
    pv1 = np.array([100.0] + [25.0] * (1 + len(zc)))
    log_ml1, focal, _ = _log_marginal_focal(X1.T @ X1, X1.T @ zy,
                                            float(zy @ zy), n, pv1, 1)
    X0 = np.column_stack([np.ones(n)] + zc)
    pv0 = np.array([100.0] + [25.0] * len(zc))
    log_ml0, _, _ = _log_marginal_focal(X0.T @ X0, X0.T @ zy,
                                        float(zy @ zy), n, pv0, None)
    m_f, var_f, w = focal
    # P(sign(beta) = direction) under the sigma-mixture posterior
    if direction > 0:
        pd_dir = float(np.sum(w * norm.cdf(m_f / np.sqrt(var_f))))
    else:
        pd_dir = float(np.sum(w * norm.cdf(-m_f / np.sqrt(var_f))))
    log_bf = float(log_ml1 - log_ml0)
    return {
        "log10_BF10": log_bf / np.log(10.0),
        "log_BF10": log_bf,
        "BF10": float(np.exp(min(log_bf, 700.0))),  # clipped; decisions need <= 10
        "beta_mean_std": float(np.sum(w * m_f)),
        "Pd": pd_dir,
        "n": int(n),
    }


def p_claim(bf10_adj, pd_adj, w, log_bf10=None):
    """Posterior claim probability, computed on the log-odds scale so that
    very large Bayes factors cannot overflow: logit(p_inc) = logit(w) +
    log BF10_adj."""
    log_bf = float(np.log(bf10_adj)) if log_bf10 is None else float(log_bf10)
    logit = np.log(w / (1.0 - w)) + log_bf
    p_inc = float(1.0 / (1.0 + np.exp(-logit))) if logit < 700 else 1.0
    return p_inc * pd_adj, p_inc


def ledger_prefix(ratified):
    """ratified: list of dicts with 'p_claim'. Returns set of indices (into the
    input list) ratified by the ledger rule: largest prefix, p_claim
    descending, with mean(1 - p_claim) <= 0.05."""
    order = sorted(range(len(ratified)), key=lambda i: -ratified[i]["p_claim"])
    keep, cum = [], 0.0
    for rank, i in enumerate(order):
        cum += 1.0 - ratified[i]["p_claim"]
        if cum / (rank + 1) <= 0.05:
            keep.append(i)
        else:
            break
    return set(keep)
