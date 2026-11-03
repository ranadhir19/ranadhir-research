"""Theory-augmented option pricing: Black-Scholes + ML residual (DSAI).

Original implementation of the core computational idea in
"Theoretical Augmentation Bridging Analytic Finance and Artificial
Intelligence for Robust Option Pricing in Extreme Market Regimes"
(paper under journal revision, 2026).

Model: C_hat = C_BS + f_theta(x), where C_BS is the analytic Black-Scholes
call price (the "theory") and f_theta is a data-driven residual correction
(here: ridge regression on option features). The paper's claim is that the
theory anchor keeps pricing robust when markets leave the training regime,
whereas a pure-ML pricer collapses.

Experiment design (synthetic panel):
- Calm regime:   price = C_BS + mild skew shape + small noise.
- Crisis regime: price = C_BS + amplified volatility-smile skew + shock,
  with a VIX-like regime indicator jumping from ~15-22 to ~30-60.
- Train ONLY on calm data; test on crisis data. Compare:
    (a) pure ML:  ridge on (K/S, T, r, sigma, VIX) predicting price;
    (b) BS-only:  analytic Black-Scholes with base volatility;
    (c) hybrid:   C_BS + ridge on the residual (price - C_BS).
  Metrics: RMSE and MAE on the crisis test set.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

__all__ = [
    "bs_call",
    "make_option_panel",
    "fit_ridge",
    "predict_ridge",
    "rmse",
    "mae",
    "run_pricing_comparison",
]

FEATURE_NAMES = ["moneyness", "maturity", "rate", "vol", "vix"]


def bs_call(
    S: np.ndarray, K: np.ndarray, T: np.ndarray,
    r: np.ndarray, sigma: np.ndarray,
) -> np.ndarray:
    """Black-Scholes European call price (vectorized, numpy only)."""
    S, K, T, r, sigma = (np.asarray(a, dtype=float) for a in (S, K, T, r, sigma))
    sigma = np.maximum(sigma, 1e-8)
    T = np.maximum(T, 1e-8)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * ndtr(d1) - K * np.exp(-r * T) * ndtr(d2)


def _skew_shape(moneyness: np.ndarray, vix: np.ndarray) -> np.ndarray:
    """Volatility-smile adjustment shape (mild in calm, amplified in crisis)."""
    return (30.0 * (moneyness - 1.0) ** 2 * (vix / 35.0)
            + 0.15 * np.maximum(vix - 25.0, 0.0))


def make_option_panel(
    n: int = 4000, regime: str = "calm", seed: int = 0
) -> dict:
    """Synthetic option panel for one market regime.

    Calm:   price = C_BS + 0.2 * skew_shape + N(0, 0.25), VIX in [12, 22].
    Crisis: price = C_BS + 1.0 * skew_shape + N(0, 0.50), VIX in [30, 60].
    The same skew *shape* appears in both regimes; only its amplitude and
    the VIX range change -- the structure a residual learner can transfer.
    """
    rng = np.random.default_rng(seed)
    S = rng.lognormal(np.log(100.0), 0.15, size=n)
    moneyness = rng.uniform(0.8, 1.2, size=n)
    K = moneyness * S
    T = rng.uniform(0.1, 2.0, size=n)
    r = rng.uniform(0.02, 0.05, size=n)
    sigma = rng.uniform(0.15, 0.40, size=n)
    if regime == "calm":
        vix = rng.uniform(12.0, 22.0, size=n)
        amp, noise = 0.2, 0.25
    elif regime == "crisis":
        vix = rng.uniform(30.0, 60.0, size=n)
        amp, noise = 1.0, 0.50
    else:
        raise ValueError("regime must be 'calm' or 'crisis'")

    c_bs = bs_call(S, K, T, r, sigma)
    skew = _skew_shape(moneyness, vix)
    price = c_bs + amp * skew + rng.normal(0.0, noise, size=n)
    price = np.maximum(price, 0.05)
    X = np.column_stack([moneyness, T, r, sigma, vix])
    return {"X": X, "price": price, "c_bs": c_bs, "vix": vix,
            "regime": regime, "skew": skew}


def fit_ridge(
    X: np.ndarray, y: np.ndarray, lam: float = 1.0
) -> dict:
    """Ridge regression with standardized features and unpenalized intercept.

    Solved via the normal equations with lstsq for stability.
    """
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-12
    Xs = (X - mu) / sd
    A = np.column_stack([Xs, np.ones(len(Xs))])
    XtX = A.T @ A
    XtX[np.arange(A.shape[1] - 1), np.arange(A.shape[1] - 1)] += lam
    coef = np.linalg.lstsq(XtX, A.T @ y, rcond=None)[0]
    return {"coef": coef, "mu": mu, "sd": sd, "lam": lam}


def predict_ridge(model: dict, X: np.ndarray) -> np.ndarray:
    """Predict with a fitted ridge model."""
    Xs = (X - model["mu"]) / model["sd"]
    A = np.column_stack([Xs, np.ones(len(Xs))])
    return A @ model["coef"]


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y - yhat)))


def run_pricing_comparison(
    n_train: int = 4000,
    n_test: int = 4000,
    lam: float = 1.0,
    seed: int = 0,
) -> dict:
    """Train on calm, test on crisis: pure ML vs BS-only vs hybrid.

    Returns RMSE/MAE per model on the crisis test set, plus the calm-set
    training RMSE for context.
    """
    train = make_option_panel(n_train, "calm", seed=seed)
    test = make_option_panel(n_test, "crisis", seed=seed + 999)

    # (a) pure ML: ridge directly on price
    ml_model = fit_ridge(train["X"], train["price"], lam=lam)
    ml_pred = predict_ridge(ml_model, test["X"])

    # (b) theory only: analytic Black-Scholes
    bs_pred = test["c_bs"]

    # (c) hybrid: C_BS + ridge on the residual
    resid = train["price"] - train["c_bs"]
    res_model = fit_ridge(train["X"], resid, lam=lam)
    hybrid_pred = test["c_bs"] + predict_ridge(res_model, test["X"])

    y = test["price"]
    return {
        "pure_ml": {"rmse": rmse(y, ml_pred), "mae": mae(y, ml_pred)},
        "bs_only": {"rmse": rmse(y, bs_pred), "mae": mae(y, bs_pred)},
        "hybrid": {"rmse": rmse(y, hybrid_pred), "mae": mae(y, hybrid_pred)},
        "train_rmse_pure_ml": rmse(train["price"],
                                   predict_ridge(ml_model, train["X"])),
        "train_rmse_hybrid": rmse(train["price"],
                                  train["c_bs"] + predict_ridge(res_model,
                                                                 train["X"])),
        "n_train": n_train,
        "n_test": n_test,
    }
