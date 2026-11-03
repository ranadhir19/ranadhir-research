"""Siamese contrastive verification for bank-cheque CAR/LAR amounts.

A cheque carries its amount twice: the CAR ("courtesy amount recognition",
e.g. ``$125.00``) and the LAR ("legal amount recognition", e.g. "one
hundred twenty five dollars").  Verification is framed as pair
classification: given (CAR, LAR), do they encode the same amount?

Pipeline (all numpy):
- synthetic pair generator: amounts -> CAR/LAR strings; negatives are
  mismatched amounts;
- bigram-hash character features (256-d signed) per string; the pair is
  represented by ``|f_car - f_lar|``;
- a 16-d *linear* projection trained with contrastive loss (margin 1.0)
  by SGD;
- baseline: raw ``|f_car - f_lar|`` distance with a tuned threshold.

The honest finding (the paper's own): the linear embedding barely beats
chance, showing linear similarity on hash features is insufficient and
motivating deep encoders.

This is an original implementation of the paper's core idea, written from
the method description; the paper is under journal revision (2026).
"""

from __future__ import annotations

import hashlib

import numpy as np

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
         "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]


def num_words(n: int) -> str:
    """English words for an integer 0 <= n <= 9999."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else "-" + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " " + num_words(rest))
    rest = n % 1000
    return _ONES[n // 1000] + " thousand" + ("" if rest == 0 else " " + num_words(rest))


def make_car(dollars: int, cents: int) -> str:
    """Courtesy amount: ``$125.00``."""
    return f"${dollars}.{cents:02d}"


def make_lar(dollars: int, cents: int) -> str:
    """Legal amount: ``one hundred twenty five dollars and thirty cents``."""
    s = f"{num_words(dollars)} dollar{'s' if dollars != 1 else ''}"
    if cents:
        s += f" and {num_words(cents)} cent{'s' if cents != 1 else ''}"
    return s


def bigram_features(s: str, dim: int = 256) -> np.ndarray:
    """Signed bigram-hash character features of a string."""
    f = np.zeros(dim)
    t = f" {s.lower()} "
    for j in range(len(t) - 1):
        h = int(hashlib.md5(t[j:j + 2].encode()).hexdigest(), 16)
        f[h % dim] += 1.0 if (h >> 7) % 2 == 0 else -1.0
    return f


def pair_vector(car: str, lar: str, dim: int = 256) -> np.ndarray:
    """Pair representation ``|f_car - f_lar|``."""
    return np.abs(bigram_features(car, dim) - bigram_features(lar, dim))


def generate_pairs(rng: np.random.Generator, n_pairs: int,
                   dim: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic (CAR, LAR) pairs: half matches, half mismatched amounts.

    Negatives are *hard*: the mismatched amount has the same dollar digit
    count as the CAR amount, removing the trivial magnitude shortcut.
    """
    X = np.zeros((n_pairs, dim))
    y = np.zeros(n_pairs, dtype=int)
    for i in range(n_pairs):
        d1, c1 = int(rng.integers(1, 10000)), int(rng.integers(0, 100))
        nd = len(str(d1))
        if i % 2 == 0:  # match
            d2, c2, y[i] = d1, c1, 1
        else:  # mismatch: same magnitude, different amount
            d2, c2 = d1, c1
            while (d2, c2) == (d1, c1):
                d2 = int(rng.integers(10 ** (nd - 1), 10 ** nd))
                c2 = int(rng.integers(0, 100))
        X[i] = pair_vector(make_car(d1, c1), make_lar(d2, c2), dim)
    return X, y


def train_projection(X: np.ndarray, y: np.ndarray, rng: np.random.Generator,
                     out_dim: int = 16, margin: float = 1.0,
                     epochs: int = 30, lr: float = 2e-3) -> np.ndarray:
    """Train the linear projection ``W`` with contrastive loss by SGD.

    Loss: ``y * dist^2 + (1 - y) * max(0, margin - dist)^2``,
    ``dist = ||X_i @ W||_2``.
    """
    n, dim = X.shape
    W = rng.normal(0.0, 0.01, (dim, out_dim))
    for _ in range(epochs):
        idx = rng.permutation(n)
        for i in idx:
            d = X[i]
            e = d @ W
            dist = float(np.linalg.norm(e))
            if y[i] == 1:
                grad = 2.0 * np.outer(d, e)
            elif dist < margin and dist > 1e-12:
                grad = -2.0 * (margin - dist) * np.outer(d, e / dist)
            else:
                continue
            W -= lr * grad
    return W


def embed_distances(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Embedding distances ``||X @ W||_2`` per pair."""
    return np.linalg.norm(X @ W, axis=1)


def accuracy_at_threshold(dists: np.ndarray, y: np.ndarray, thr: float) -> float:
    """Pair-verification accuracy: predict match iff distance < threshold."""
    return float((((dists < thr).astype(int)) == y).mean())


def best_threshold(dists: np.ndarray, y: np.ndarray, n_grid: int = 200) -> float:
    """Threshold maximizing accuracy on the given (train) pairs."""
    grid = np.linspace(dists.min(), dists.max(), n_grid)
    accs = [accuracy_at_threshold(dists, y, t) for t in grid]
    return float(grid[int(np.argmax(accs))])


def train_linear_classifier(X: np.ndarray, y: np.ndarray, rng: np.random.Generator,
                            epochs: int = 60, lr: float = 0.1,
                            batch: int = 256, l2: float = 1e-4) -> tuple[np.ndarray, float]:
    """Purely linear baseline: logistic regression on the pair vectors."""
    w = np.zeros(X.shape[1])
    b = 0.0
    n = X.shape[0]
    for _ in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch):
            bb = idx[s:s + batch]
            p = 1.0 / (1.0 + np.exp(-(X[bb] @ w + b)))
            err = p - y[bb]
            w -= lr * (X[bb].T @ err / bb.size + l2 * w)
            b -= lr * float(err.mean())
    return w, b


def linear_predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Binary predictions from a linear classifier."""
    return (1.0 / (1.0 + np.exp(-(X @ w + b))) > 0.5).astype(int)
