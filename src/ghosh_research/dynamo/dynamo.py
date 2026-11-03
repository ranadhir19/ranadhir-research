"""Contractive world-model operator verification (DYNAMO).

Implements the corrected contraction theorem for the world-model update
operator

    T(W) = (1 - alpha) * W + alpha * U(W),

where ``U`` is an MLP-style update whose Lipschitz constant ``L < 1`` is
enforced by spectrally clipping every layer: each layer's largest singular
value is capped at ``L ** (1 / d)`` for ``d`` layers, so the composed update
satisfies ``Lip(U) <= L``.  The theorem then predicts the per-step
contraction factor ``rho = (1 - alpha) + alpha * L``.

This is an original implementation of the paper's core idea, written from
the theorem statement; the paper is under journal revision (2026).
"""

from __future__ import annotations

import numpy as np


def spectrally_clipped_matrix(rng: np.random.Generator, n: int, cap: float) -> np.ndarray:
    """Draw a random ``n x n`` matrix with largest singular value <= ``cap``.

    A Gaussian matrix is decomposed via SVD and its singular values are
    clipped to ``cap``, so ``||A||_2 <= cap`` by construction.
    """
    A = rng.standard_normal((n, n))
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s = np.minimum(s, cap)
    return U @ np.diag(s) @ Vt


def build_update_operator(rng: np.random.Generator, n: int, L: float, d: int = 3) -> np.ndarray:
    """Build the MLP-style update matrix ``P`` with ``||P||_2 <= L``.

    ``P = A_d ... A_2 A_1`` where each factor is spectrally clipped to
    ``L ** (1 / d)``, so the product's spectral norm is at most ``L``.
    The update map is ``U(W) = P @ W``, whose Lipschitz constant (in
    Frobenius norm) is ``||P||_2 <= L``.
    """
    cap = L ** (1.0 / d)
    P = np.eye(n)
    for _ in range(d):
        P = spectrally_clipped_matrix(rng, n, cap) @ P
    return P


def contractive_step(W: np.ndarray, P: np.ndarray, alpha: float) -> np.ndarray:
    """Apply one world-model update ``T(W) = (1 - alpha) W + alpha P W``."""
    return (1.0 - alpha) * W + alpha * (P @ W)


def predicted_rho(alpha: float, L: float) -> float:
    """Theorem prediction: per-step contraction factor ``(1 - alpha) + alpha * L``."""
    return (1.0 - alpha) + alpha * L


def empirical_contraction(
    rng: np.random.Generator,
    P: np.ndarray,
    alpha: float,
    n_pairs: int = 8,
    n_steps: int = 50,
    n: int = 64,
    sep: float = 1e-3,
) -> tuple[float, float]:
    """Measure the per-step contraction of ``T`` between nearby starts.

    For ``n_pairs`` random nearby pairs ``(W, W')``, iterates ``T`` and
    records ``||T(W_k) - T(W'_k)||_F / ||W_k - W'_k||_F`` at each step.
    Returns ``(mean, max)`` of the per-step geometric-mean ratios.
    """
    Q = (1.0 - alpha) * np.eye(n) + alpha * P
    ratios = []
    for _ in range(n_pairs):
        W = rng.standard_normal((n, n))
        D = rng.standard_normal((n, n))
        D = sep * D / np.linalg.norm(D, "fro")
        Wp = W + D
        step_ratios = []
        for _ in range(n_steps):
            num = np.linalg.norm(Q @ (W - Wp), "fro")
            den = np.linalg.norm(W - Wp, "fro")
            if den == 0.0:
                break
            step_ratios.append(num / den)
            W, Wp = contractive_step(W, P, alpha), contractive_step(Wp, P, alpha)
        ratios.append(float(np.exp(np.mean(np.log(step_ratios)))))
    return float(np.mean(ratios)), float(np.max(ratios))


def convergence_run(
    rng: np.random.Generator,
    P: np.ndarray,
    alpha: float,
    n_steps: int = 50,
    n: int = 64,
    tol: float = 1e-3,
) -> tuple[np.ndarray, bool]:
    """Iterate ``T`` from a random start; return relative-norm trajectory and a
    flag for whether ``||W_k|| / ||W_0|| < tol`` was reached within ``n_steps``."""
    W = rng.standard_normal((n, n))
    norm0 = np.linalg.norm(W, "fro")
    traj = np.empty(n_steps + 1)
    traj[0] = 1.0
    for k in range(1, n_steps + 1):
        W = contractive_step(W, P, alpha)
        traj[k] = np.linalg.norm(W, "fro") / norm0
    return traj, bool(traj[-1] < tol)
