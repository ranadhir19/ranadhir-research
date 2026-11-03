"""Multi-Agent Memento: continual learning and the retention/plasticity trade-off.

A synthetic binary stream (``n = 2000``) whose class boundary rotates at
``t = 700`` and ``t = 1400`` (45 -> 90 -> 135 degrees) is learned two ways:

(a) Sliding-window logistic regression refit by mini-batch SGD in numpy,
    warm-started from the previous weights (the "fine-tune" analogue):
    plastic, tracks the current distribution, forgets the old ones.

(b) A growing prototype bank: every block stores the per-class feature
    means and nothing is ever deleted (the "no forgetting" analogue).
    Classification is a similarity-weighted vote,
    ``argmax_c x . M_c`` with ``M_c`` the mean of the stored class-``c``
    prototypes.  Stale prototypes are never removed, so they keep voting.

This is an original implementation of the paper's core idea, written from
the method description; the paper is under journal revision (2026).
"""

from __future__ import annotations

import numpy as np

SEGMENTS = [(0, 700), (700, 1400), (1400, 2000)]
ANGLES = (45.0, 90.0, 135.0)


def generate_stream(rng: np.random.Generator, n: int = 2000,
                    noise: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Binary stream on ``[-1, 1]^2``; the true boundary ``x2 = m x1`` rotates
    through 45/90/135 degrees at t = 700/1400, with ``noise`` label flips."""
    X = rng.uniform(-1.0, 1.0, (n, 2))
    theta = np.where(np.arange(n) < 700, ANGLES[0],
                     np.where(np.arange(n) < 1400, ANGLES[1], ANGLES[2]))
    m = np.tan(np.deg2rad(theta))
    y = (X[:, 1] > m * X[:, 0]).astype(int)
    flip = rng.random(n) < noise
    return X, np.where(flip, 1 - y, y)


def sgd_logistic_fit(X: np.ndarray, y: np.ndarray, w_init: np.ndarray,
                     rng: np.random.Generator, epochs: int = 25,
                     lr: float = 0.3, batch: int = 64, l2: float = 1e-4) -> np.ndarray:
    """Mini-batch SGD for binary logistic regression with intercept.

    ``X`` is ``(n, 2)``; ``w`` is ``(3,)`` with the last entry the intercept.
    """
    w = w_init.copy()
    Xa = np.hstack([X, np.ones((X.shape[0], 1))])
    n = X.shape[0]
    for _ in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch):
            b = idx[s:s + batch]
            p = 1.0 / (1.0 + np.exp(-Xa[b] @ w))
            grad = Xa[b].T @ (p - y[b]) / b.size + l2 * w
            w -= lr * grad
    return w


def sliding_window_refit(X: np.ndarray, y: np.ndarray, rng: np.random.Generator,
                         window: int = 400, step: int = 200) -> np.ndarray:
    """Refit logistic regression on the last ``window`` samples every ``step``
    samples, warm-starting from the previous weights.  Returns final weights."""
    w = np.zeros(3)
    for start in range(0, X.shape[0], step):
        lo = max(0, start + step - window)
        w = sgd_logistic_fit(X[lo:start + step], y[lo:start + step], w, rng)
    return w


def logistic_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Binary predictions from logistic weights ``w``."""
    Xa = np.hstack([X, np.ones((X.shape[0], 1))])
    return (1.0 / (1.0 + np.exp(-Xa @ w)) > 0.5).astype(int)


def prototype_bank_fit(X: np.ndarray, y: np.ndarray,
                       block: int = 200) -> dict[int, list[np.ndarray]]:
    """Grow the prototype bank: per block, store each class's feature mean.

    Prototypes are appended and never deleted -- the "no forgetting" analogue.
    """
    bank: dict[int, list[np.ndarray]] = {0: [], 1: []}
    for start in range(0, X.shape[0], block):
        xb, yb = X[start:start + block], y[start:start + block]
        for c in (0, 1):
            if np.any(yb == c):
                bank[c].append(xb[yb == c].mean(axis=0))
    return bank


def prototype_predict(X: np.ndarray, bank: dict[int, list[np.ndarray]]) -> np.ndarray:
    """Similarity-weighted vote: ``argmax_c x . M_c`` where ``M_c`` is the mean
    of the stored class-``c`` prototypes."""
    M = {c: np.mean(bank[c], axis=0) for c in (0, 1)}
    return (X @ M[1] > X @ M[0]).astype(int)


def segment_accuracies(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Accuracy overall and per segment ``0-700 / 700-1400 / 1400-2000``."""
    out = {"overall": float((pred == y).mean())}
    for (a, b) in SEGMENTS:
        out[f"{a}-{b}"] = float((pred[a:b] == y[a:b]).mean())
    return out
