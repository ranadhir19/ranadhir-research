"""Greedy maximal ethical subset construction (Coarse Set Theory).

Original implementation of the core computational idea in
"Coarse Set Theory for Ethical AI: A Mathematical Framework for
Multi-Constraint Decision Making with Formal Guarantees"
(paper under journal revision, 2026).

The paper treats ethical decision-making as the construction of a *maximal*
subset M of candidate decisions such that M jointly satisfies a family of
ethical constraints. The constructive procedure is greedy:

    M = empty; candidates = X
    while candidates nonempty:
        x* = argmin over candidates of  sum_i (1 - I^{c_i}(x))^2
        # I^{c_i}(x) = 1 if admitting x keeps constraint c_i satisfied
        if satisfaction(x*) >= 1 - eps: admit x*, drop its eps-neighbours
        else: discard x*

Here the constraints are group-fairness bounds on a lending-style
classifier: the demographic-parity gap and the equal-opportunity gap of the
admitted decision set must each stay below a tolerance tau. Constraint
indicators are evaluated on the *hypothetical* admitted set M U {x}, with
running per-group counts so each iteration is O(#candidates). Ties in the
argmin are broken by descending decision utility (predicted score), which
preserves accuracy while the constraints bind.

This is a faithful computational rendering of the paper's construction, not
a copy of any code from the paper (none was published with it).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "sigmoid",
    "make_synthetic_lending",
    "tune_threshold",
    "demographic_parity_gap",
    "equalized_opportunity_gap",
    "accuracy",
    "greedy_ethical_subset",
    "subset_report",
]

N_CONSTRAINTS = 2


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic sigmoid, elementwise."""
    return 1.0 / (1.0 + np.exp(-z))


def make_synthetic_lending(
    n: int = 2000,
    n_features: int = 3,
    group_rate: float = 0.4,
    group_shift: float = 0.4,
    score_noise: float = 0.4,
    seed: int = 0,
) -> dict:
    """Generate a synthetic lending population with a biased score.

    Features X are standard normal; a binary group attribute G shifts the
    latent creditworthiness upward for group 1 (the advantaged group), which
    makes a globally-tuned threshold classifier violate demographic parity.
    The default ``group_shift`` reproduces the paper's reported baseline
    gap scale (DP gap ~= 0.12). The observed ``scores`` add estimation noise
    so the baseline classifier is imperfect and there is accuracy worth
    trading off.

    Returns a dict with ``features``, ``group``, ``labels``, ``scores``.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, n_features))
    group = (rng.random(n) < group_rate).astype(int)
    w = np.array([0.8, 0.5, -0.6][:n_features], dtype=float)
    z = X @ w + group_shift * group - 0.1
    true_prob = sigmoid(1.1 * z)
    labels = (rng.random(n) < true_prob).astype(int)
    scores = sigmoid(1.1 * z + rng.normal(0.0, score_noise, size=n))
    return {"features": X, "group": group, "labels": labels, "scores": scores}


def tune_threshold(scores: np.ndarray, labels: np.ndarray, grid: int = 200) -> float:
    """Pick the global score threshold maximizing training accuracy."""
    thresholds = np.linspace(scores.min(), scores.max(), grid)
    accs = np.array([accuracy((scores >= t).astype(int), labels) for t in thresholds])
    return float(thresholds[int(np.argmax(accs))])


def accuracy(pred: np.ndarray, labels: np.ndarray) -> float:
    """Classification accuracy."""
    return float(np.mean(pred == labels))


def demographic_parity_gap(pred: np.ndarray, group: np.ndarray) -> float:
    """|P(yhat=1 | g=0) - P(yhat=1 | g=1)|."""
    r0 = pred[group == 0]
    r1 = pred[group == 1]
    if len(r0) == 0 or len(r1) == 0:
        return 0.0
    return float(abs(r0.mean() - r1.mean()))


def equalized_opportunity_gap(
    pred: np.ndarray, labels: np.ndarray, group: np.ndarray
) -> float:
    """|TPR(g=0) - TPR(g=1)| over the truly-positive subpopulations."""
    gaps = []
    for g in (0, 1):
        mask = (group == g) & (labels == 1)
        gaps.append(float(pred[mask].mean()) if mask.sum() else 0.0)
    return abs(gaps[0] - gaps[1])


def _hypothetical_gaps(
    n0: int, pos0: float, y1_0: float, tp0: float,
    n1: int, pos1: float, y1_1: float, tp1: float,
    g: np.ndarray, p: np.ndarray, lab: np.ndarray,
    tau_dp: float, tau_eo: float,
) -> np.ndarray:
    """Violation count v(x) = sum_i (1 - I^{c_i}(x))^2 for every candidate.

    Constraint indicators are evaluated on the hypothetical admitted set
    M U {x} using the current running per-group counts plus each candidate's
    own contribution (fully vectorized over candidates).
    """
    g0 = (g == 0).astype(float)
    g1 = (g == 1).astype(float)
    N0 = n0 + g0
    N1 = n1 + g1
    P0 = pos0 + p * g0
    P1 = pos1 + p * g1
    Y0 = y1_0 + lab * g0
    Y1 = y1_1 + lab * g1
    T0 = tp0 + p * lab * g0
    T1 = tp1 + p * lab * g1
    dp = np.abs(np.divide(P0, N0, out=np.zeros_like(P0), where=N0 > 0)
                - np.divide(P1, N1, out=np.zeros_like(P1), where=N1 > 0))
    eo = np.abs(np.divide(T0, Y0, out=np.zeros_like(T0), where=Y0 > 0)
                - np.divide(T1, Y1, out=np.zeros_like(T1), where=Y1 > 0))
    return (dp > tau_dp).astype(int) + (eo > tau_eo).astype(int)


def greedy_ethical_subset(
    features: np.ndarray,
    pred: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    group: np.ndarray,
    tau_dp: float = 0.015,
    tau_eo: float = 0.015,
    eps: float = 0.0,
    delta: float = 0.35,
    seed: int = 0,
) -> dict:
    """Greedy construction of a maximal ethically-admissible decision set.

    Each iteration picks ``x* = argmin_x sum_i (1 - I^{c_i}(x))^2`` (ties
    broken by descending utility ``scores``), admits x* when its constraint
    satisfaction ``1 - v(x*)/m`` is at least ``1 - eps``, and then removes
    all remaining candidates within feature distance ``delta`` of x* (the
    eps-proximity neighbourhood, which enforces spread/maximality).
    Candidates that would violate a constraint are discarded.

    Returns a dict with the boolean ``admitted`` mask and run statistics.
    """
    rng = np.random.default_rng(seed)  # reserved for future stochastic tie-breaks
    _ = rng
    n = len(pred)
    mu = features.mean(axis=0)
    sd = features.std(axis=0) + 1e-12
    Z = (features - mu) / sd

    candidates = np.ones(n, dtype=bool)
    admitted = np.zeros(n, dtype=bool)
    # running per-group counts over the admitted set M
    n0 = n1 = 0
    pos0 = pos1 = 0.0
    y1_0 = y1_1 = 0.0
    tp0 = tp1 = 0.0
    n_discarded = 0

    while candidates.any():
        c = np.flatnonzero(candidates)
        v = _hypothetical_gaps(
            n0, pos0, y1_0, tp0, n1, pos1, y1_1, tp1,
            group[c], pred[c].astype(float), labels[c].astype(float),
            tau_dp, tau_eo,
        )
        # argmin of violation; tie-break by descending utility (accuracy-preserving)
        order = np.lexsort((-scores[c], v))
        star = c[order[0]]
        satisfaction = 1.0 - v[order[0]] / N_CONSTRAINTS
        if satisfaction >= 1.0 - eps:
            admitted[star] = True
            g, p, lab = group[star], float(pred[star]), float(labels[star])
            if g == 0:
                n0 += 1; pos0 += p; y1_0 += lab; tp0 += p * lab
            else:
                n1 += 1; pos1 += p; y1_1 += lab; tp1 += p * lab
            dists = np.linalg.norm(Z[candidates] - Z[star], axis=1)
            neighbours = np.flatnonzero(candidates)[dists < delta]
            candidates[neighbours] = False
        else:
            candidates[star] = False
            n_discarded += 1

    return {
        "admitted": admitted,
        "n_admitted": int(admitted.sum()),
        "n_discarded": n_discarded,
        "admission_rate": float(admitted.mean()),
    }


def subset_report(
    pred: np.ndarray,
    labels: np.ndarray,
    group: np.ndarray,
    admitted: np.ndarray,
) -> dict:
    """Fairness/accuracy metrics on the full set vs the admitted subset."""
    return {
        "dp_gap_full": demographic_parity_gap(pred, group),
        "eo_gap_full": equalized_opportunity_gap(pred, labels, group),
        "accuracy_full": accuracy(pred, labels),
        "dp_gap_admitted": demographic_parity_gap(pred[admitted], group[admitted]),
        "eo_gap_admitted": equalized_opportunity_gap(
            pred[admitted], labels[admitted], group[admitted]
        ),
        "accuracy_admitted": accuracy(pred[admitted], labels[admitted]),
        "n_admitted": int(admitted.sum()),
        "n_total": len(pred),
    }
