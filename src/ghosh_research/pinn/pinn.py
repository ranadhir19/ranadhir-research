"""Hard-negative stress test + Flory-Huggins physics loss (PINN for LLPS).

Original implementation of the core computational ideas in
"Physics-Informed Neural Networks for Predicting Liquid-Liquid Phase
Separation in Neurodegenerative Disease Proteins" (paper under journal
revision, 2026).

Two components:

1. Hard-negative stress test. Synthetic protein panels described by four
   biophysical features (disorder content, charge-patterning kappa,
   aromatic fraction, hydropathy). Phase-separating ("positive") proteins
   come from one region of feature space; "easy" negatives are drawn far
   away, while "hard" negatives overlap the positive region. A numpy
   logistic-regression classifier (L-BFGS via scipy) is evaluated with
   stratified 5-fold cross-validation under three protocols:
     P1: positives vs easy negatives   (expect AUC ~= 1.000)
     P2: positives vs balanced easy+hard mix
     P3: positives vs hard negatives only (expect AUC ~= 0.91)
   This reproduces the paper's stress-test finding: headline accuracy on
   easy negatives collapses once the classifier faces hard negatives.

2. Flory-Huggins physics-informed loss (demo). The paper regularizes its
   network with a physics residual derived from Flory-Huggins theory.
   Here we implement the loss specification exactly --
     L = BCE + lambda1 * |dG_FH/dphi|^2 + lambda2 * (p_perc - p_c)^2
   with G_FH(phi) = phi log phi + (1-phi) log(1-phi) + chi phi (1-phi) --
   and show each term is computable on sample composition points. This is
   an honest loss-function demonstration, not a trained network.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

__all__ = [
    "FEATURE_NAMES",
    "make_protein_panel",
    "fit_logreg",
    "predict_proba",
    "auc_score",
    "stratified_kfold",
    "protocol_auc",
    "flory_huggins",
    "d_flory_huggins",
    "fhp_loss_demo",
]

FEATURE_NAMES = ["disorder", "kappa", "aromatic", "hydropathy"]


def make_protein_panel(
    n_pos: int = 400,
    n_easy: int = 400,
    n_hard: int = 400,
    seed: int = 0,
) -> dict:
    """Synthetic protein panel with easy and hard negatives.

    Positives: high disorder, mid kappa, elevated aromatic fraction, low
    hydropathy (LLPS-prone region). Easy negatives: the opposite corner.
    Hard negatives: overlapping the positive region (same means shifted by
    less than one standard deviation on each feature).
    """
    rng = np.random.default_rng(seed)

    def block(n, means, sds):
        X = rng.normal(np.array(means), np.array(sds), size=(n, 4))
        return np.clip(X, 0.0, 1.0)

    X_pos = block(n_pos, [0.68, 0.42, 0.130, 0.35],
                  [0.07, 0.09, 0.025, 0.09])
    X_easy = block(n_easy, [0.15, 0.75, 0.030, 0.75],
                   [0.06, 0.10, 0.015, 0.10])
    X_hard = block(n_hard, [0.59, 0.50, 0.107, 0.43],
                   [0.09, 0.10, 0.025, 0.10])
    X = np.vstack([X_pos, X_easy, X_hard])
    y = np.concatenate([np.ones(n_pos), np.zeros(n_easy + n_hard)]).astype(int)
    kind = (["pos"] * n_pos + ["easy"] * n_easy + ["hard"] * n_hard)
    perm = rng.permutation(len(X))
    return {"X": X[perm], "y": y[perm],
            "kind": np.array(kind)[perm]}


def fit_logreg(
    X: np.ndarray, y: np.ndarray, l2: float = 1e-3, max_iter: int = 500
) -> dict:
    """Fit L2-regularized logistic regression (numpy + scipy L-BFGS).

    Features are standardized internally; the returned model stores the
    scaler so ``predict_proba`` can be called on raw features.
    """
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-12
    Xs = (X - mu) / sd
    n, p = Xs.shape

    def loss(theta):
        w, b = theta[:p], theta[p]
        z = Xs @ w + b
        # numerically stable BCE
        bce = np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z))))
        return bce + 0.5 * l2 * float(w @ w)

    res = minimize(loss, np.zeros(p + 1), method="L-BFGS-B",
                   options={"maxiter": max_iter})
    return {"w": res.x[:p], "b": float(res.x[p]), "mu": mu, "sd": sd,
            "success": bool(res.success)}


def predict_proba(model: dict, X: np.ndarray) -> np.ndarray:
    """Positive-class probabilities from a fitted model."""
    Xs = (X - model["mu"]) / model["sd"]
    return 1.0 / (1.0 + np.exp(-(Xs @ model["w"] + model["b"])))


def auc_score(y: np.ndarray, s: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney U statistic (pure numpy)."""
    y = np.asarray(y, dtype=int)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    u = ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0
    return float(u / (n1 * n0))


def stratified_kfold(
    y: np.ndarray, k: int = 5, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified k-fold train/test index splits (deterministic)."""
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        for i, j in enumerate(idx):
            folds[i % k].append(int(j))
    idx_all = np.arange(len(y))
    return [(np.setdiff1d(idx_all, np.array(f)),
             np.array(f)) for f in folds]


def protocol_auc(
    X: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 0, l2: float = 1e-3
) -> dict:
    """Pooled out-of-fold AUC under stratified k-fold evaluation."""
    oof = np.zeros(len(y))
    for tr, te in stratified_kfold(y, k, seed):
        model = fit_logreg(X[tr], y[tr], l2=l2)
        oof[te] = predict_proba(model, X[te])
    return {"auc": auc_score(y, oof), "oof_proba": oof, "k": k}


def flory_huggins(phi: np.ndarray, chi: float) -> np.ndarray:
    """Flory-Huggins mixing free energy G_FH(phi).

    G_FH(phi) = phi log phi + (1 - phi) log(1 - phi) + chi phi (1 - phi).
    """
    phi = np.clip(np.asarray(phi, dtype=float), 1e-12, 1 - 1e-12)
    return (phi * np.log(phi) + (1 - phi) * np.log(1 - phi)
            + chi * phi * (1 - phi))


def d_flory_huggins(phi: np.ndarray, chi: float) -> np.ndarray:
    """Analytic derivative dG_FH/dphi = log(phi/(1-phi)) + chi (1 - 2 phi)."""
    phi = np.clip(np.asarray(phi, dtype=float), 1e-12, 1 - 1e-12)
    return np.log(phi / (1 - phi)) + chi * (1 - 2 * phi)


def fhp_loss_demo(
    phi: np.ndarray,
    y: np.ndarray,
    p_pred: np.ndarray,
    chi: float = 2.5,
    lam1: float = 0.1,
    lam2: float = 0.1,
    p_perc: float = 0.42,
    p_c: float = 0.30,
) -> dict:
    """Evaluate the paper's physics-informed loss spec on sample points.

    L = BCE(y, p_pred) + lam1 * |dG_FH/dphi|^2 + lam2 * (p_perc - p_c)^2.

    This demonstrates that every term of the physics-informed loss is
    computable; it is a loss-function demo, not a trained network.
    """
    eps = 1e-12
    pp = np.clip(p_pred, eps, 1 - eps)
    bce = float(np.mean(-(y * np.log(pp) + (1 - y) * np.log(1 - pp))))
    phys1 = float(np.mean(d_flory_huggins(phi, chi) ** 2))
    phys2 = float((p_perc - p_c) ** 2)
    return {
        "bce": bce,
        "physics_grad_term": lam1 * phys1,
        "physics_perc_term": lam2 * phys2,
        "total": bce + lam1 * phys1 + lam2 * phys2,
        "chi": chi, "lambda1": lam1, "lambda2": lam2,
        "note": "loss-function demo on sample points; not a trained network",
    }
