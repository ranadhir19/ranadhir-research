#!/usr/bin/env python3
"""Paper 7 upgrade cycle 2: method-of-moments reliability-corrected
adjusted computation (Component C). Protocol frozen in
../upgrade2_preregistration.md Section 1.

Identical model, priors, standardisation and log-sigma quadrature to
cycle-1 adjusted_core.py. The ONLY change: the Gram matrix X'X (on
standardised variables) has the diagonal entries of the covariate
columns multiplied by lambda in (0,1] (in BOTH alternative and null);
X'y and y'y are unchanged, per the classical-error identities
Cov(z*, x) = Cov(z, x), Cov(z*, y) = Cov(z, y). The result is a
method-of-moments corrected Bayes factor: an approximation to the
errors-in-variables posterior, NOT the exact latent-z posterior.
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
    """Same formulas as adjusted_core._log_marginal_focal. Returns None
    if the posterior precision is not positive definite at any grid
    sigma (the frozen not_evaluable fallback)."""
    p = XtX.shape[0]
    lam_prec = 1.0 / prior_var
    s2 = SIG_GRID ** 2
    A = XtX[None, :, :] / s2[:, None, None] + np.diag(lam_prec)[None, :, :]
    try:
        L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        return None
    logdetA = 2.0 * np.log(np.diagonal(L, axis1=1, axis2=2)).sum(axis=1)
    rhs = Xty[None, :] / s2[:, None]
    m = np.linalg.solve(A, rhs[:, :, None])[:, :, 0]
    log_ml_s = (-0.5 * n * LN2PI - n * np.log(SIG_GRID)
                - 0.5 * float(np.log(prior_var).sum())
                - 0.5 * logdetA
                - 0.5 * (yty / s2 - np.einsum('gi,gi->g', m, rhs)))
    log_integrand = (log_ml_s + 0.5 * np.log(2.0 / np.pi)
                     - 0.5 * s2 + np.log(SIG_GRID))
    mx = float(log_integrand.max())
    w_unnorm = TRAP_W * np.exp(log_integrand - mx)
    log_ml = mx + float(np.log(w_unnorm.sum()))
    w = w_unnorm / w_unnorm.sum()
    if focal_idx is None:
        return log_ml, None
    ef = np.zeros((len(SIG_GRID), p, 1)); ef[:, focal_idx, 0] = 1.0
    Ainv_ef = np.linalg.solve(A, ef)[:, :, 0]
    var_f = Ainv_ef[:, focal_idx]
    return log_ml, (m[:, focal_idx], var_f, w)


def _correct_gram(XtX, cov_col_idx, lam):
    G = XtX.copy()
    for j in cov_col_idx:
        G[j, j] *= lam
    return G


def fit_corrected(y, x_focal, covs, direction, lam):
    """Corrected analogue of adjusted_core.fit_adjusted. covs may be [].
    Returns dict; on not_evaluable, 'evaluable' is False and numeric
    fields are None."""
    y = np.asarray(y, float); x = np.asarray(x_focal, float)
    cols = [np.asarray(c, float) for c in covs]
    def z(v):
        sd = v.std()
        return (v - v.mean()) / sd if sd > 0 else v - v.mean()
    zy, zx = z(y), z(x)
    zc = [z(c) for c in cols]
    n = len(zy)
    X1 = np.column_stack([np.ones(n), zx] + zc)
    pv1 = np.array([100.0] + [25.0] * (1 + len(zc)))
    G1 = _correct_gram(X1.T @ X1, list(range(2, X1.shape[1])), lam)
    r1 = _log_marginal_focal(G1, X1.T @ zy, float(zy @ zy), n, pv1, 1)
    X0 = np.column_stack([np.ones(n)] + zc)
    pv0 = np.array([100.0] + [25.0] * len(zc))
    G0 = _correct_gram(X0.T @ X0, list(range(1, X0.shape[1])), lam)
    r0 = _log_marginal_focal(G0, X0.T @ zy, float(zy @ zy), n, pv0, None)
    if r1 is None or r0 is None:
        return {"evaluable": False, "lam": float(lam), "n": int(n),
                "log10_BF10": None, "log_BF10": None, "BF10": None,
                "beta_mean_std": None, "Pd": None}
    log_ml1, focal = r1
    log_ml0, _ = r0
    m_f, var_f, w = focal
    if direction > 0:
        pd_dir = float(np.sum(w * norm.cdf(m_f / np.sqrt(var_f))))
    else:
        pd_dir = float(np.sum(w * norm.cdf(-m_f / np.sqrt(var_f))))
    log_bf = float(log_ml1 - log_ml0)
    return {
        "evaluable": True, "lam": float(lam),
        "log10_BF10": log_bf / np.log(10.0),
        "log_BF10": log_bf,
        "BF10": float(np.exp(min(log_bf, 700.0))),
        "beta_mean_std": float(np.sum(w * m_f)),
        "Pd": pd_dir,
        "n": int(n),
    }


def decide_corrected(fit):
    """Decision rule of record applied to a corrected fit; a
    not-evaluable fit is NOT accept (frozen fallback)."""
    if not fit["evaluable"]:
        return "not_evaluable"
    bf, pd_ = fit["BF10"], fit["Pd"]
    if bf >= 10 and pd_ >= 0.95:
        return "accept"
    if 3 <= bf < 10 and pd_ >= 0.90:
        return "accept"
    if bf <= 1 / 3:
        return "reject"
    return "inconclusive"


def lambda_min(decisions_by_lam, grid=(1.0, 0.9, 0.8, 0.7, 0.6)):
    """Smallest grid lambda with ACCEPT at that lambda and at every
    larger grid lambda; None if not accept at lambda = 1.0."""
    if decisions_by_lam.get(1.0) != "accept":
        return None
    lm = 1.0
    for lam in grid:
        if decisions_by_lam.get(lam) == "accept":
            lm = lam
        else:
            break
    return lm
