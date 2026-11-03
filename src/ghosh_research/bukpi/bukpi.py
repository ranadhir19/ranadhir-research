"""Type-aware bottom-up discovery pipeline for cross-domain enterprise KPIs.

``correlate(x, y)`` is a type-aware facade: it inspects the column types of
its two inputs and dispatches to the association measure suited to that
type pair, so KPI relationships can be discovered bottom-up from raw
enterprise tables without a pre-declared schema:

- numeric  x numeric      -> Pearson r
- numeric  x categorical  -> ANOVA eta^2 (between-group SS / total SS)
- categorical x categorical -> Cramer's V (chi^2 on the contingency table)
- text x label            -> 64-d char-trigram hash sketch, Lloyd k-means
                             (k=3, numpy) on the sketches, then Cramer's V
                             between cluster ids and the label

This is an original implementation of the paper's core idea, written from
the method description; the paper is under journal revision (2026).
"""

from __future__ import annotations

import hashlib

import numpy as np
from scipy.stats import chi2_contingency


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation for two numeric columns."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xc = x - x.mean()
    yc = y - y.mean()
    den = np.sqrt(np.sum(xc ** 2) * np.sum(yc ** 2))
    return float(np.sum(xc * yc) / den) if den > 0 else 0.0


def anova_eta2(num: np.ndarray, cat: np.ndarray) -> float:
    """ANOVA eta^2 = between-group SS / total SS for numeric ~ categorical."""
    num = np.asarray(num, dtype=float)
    cat = np.asarray(cat)
    grand = num.mean()
    ss_between = 0.0
    for level in np.unique(cat):
        g = num[cat == level]
        ss_between += g.size * (g.mean() - grand) ** 2
    ss_total = np.sum((num - grand) ** 2)
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def cramers_v(a: np.ndarray, b: np.ndarray) -> float:
    """Cramer's V for two categorical columns via the chi^2 contingency table."""
    a = np.asarray(a)
    b = np.asarray(b)
    levels_a = np.unique(a)
    levels_b = np.unique(b)
    table = np.zeros((levels_a.size, levels_b.size))
    for i, la in enumerate(levels_a):
        for j, lb in enumerate(levels_b):
            table[i, j] = np.sum((a == la) & (b == lb))
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.sum()
    denom = n * min(table.shape[0] - 1, table.shape[1] - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else 0.0


def hash_sketch(texts: list[str], dim: int = 64) -> np.ndarray:
    """Feature-hashed char-trigram sketch: each text -> ``dim``-d signed vector."""
    X = np.zeros((len(texts), dim))
    for i, t in enumerate(texts):
        s = f" {t.lower()} "
        for j in range(len(s) - 2):
            tri = s[j:j + 3]
            h = int(hashlib.md5(tri.encode()).hexdigest(), 16)
            X[i, h % dim] += 1.0 if (h >> 7) % 2 == 0 else -1.0
    return X


def lloyd_kmeans(X: np.ndarray, k: int, rng: np.random.Generator,
                 max_iter: int = 100) -> np.ndarray:
    """Lloyd's k-means in numpy; returns integer cluster assignments."""
    idx = rng.choice(X.shape[0], size=k, replace=False)
    centers = X[idx].copy()
    assign = np.zeros(X.shape[0], dtype=int)
    for _ in range(max_iter):
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_assign = d2.argmin(axis=1)
        if np.array_equal(new_assign, assign):
            break
        assign = new_assign
        for c in range(k):
            if np.any(assign == c):
                centers[c] = X[assign == c].mean(axis=0)
    return assign


def text_cluster_v(texts: list[str], label: np.ndarray,
                   rng: np.random.Generator, k: int = 3, dim: int = 64) -> float:
    """Text branch: hash sketches -> k-means clusters -> Cramer's V vs label."""
    X = hash_sketch(texts, dim=dim)
    clusters = lloyd_kmeans(X, k, rng)
    return cramers_v(clusters, np.asarray(label))


def _is_numeric(col) -> bool:
    arr = np.asarray(col)
    return np.issubdtype(arr.dtype, np.number)


def correlate(x, y, rng: np.random.Generator | None = None) -> dict:
    """Type-aware association facade.

    Dispatches on column types and returns ``{"measure": name, "value": v,
    "branch": ...}``.  Text input is detected as a list/tuple of strings.
    """
    xa, ya = np.asarray(x), np.asarray(y)
    x_is_text = isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], str)
    y_is_text = isinstance(y, (list, tuple)) and len(y) > 0 and isinstance(y[0], str)

    if x_is_text or y_is_text:
        texts = list(x) if x_is_text else list(y)
        label = ya if x_is_text else xa
        if rng is None:
            rng = np.random.default_rng(0)
        v = text_cluster_v(texts, label, rng)
        return {"measure": "cramers_v(text_clusters, label)", "value": v, "branch": "text"}
    xn, yn = _is_numeric(xa), _is_numeric(ya)
    if xn and yn:
        return {"measure": "pearson_r", "value": pearson_r(xa, ya), "branch": "numeric-numeric"}
    if xn and not yn:
        return {"measure": "anova_eta2", "value": anova_eta2(xa, ya), "branch": "numeric-categorical"}
    if yn and not xn:
        return {"measure": "anova_eta2", "value": anova_eta2(ya, xa), "branch": "numeric-categorical"}
    return {"measure": "cramers_v", "value": cramers_v(xa, ya), "branch": "categorical-categorical"}
