"""Quickstart: DYNAMO contractive world-model operator (runs in seconds).

Builds the spectrally clipped update U with Lipschitz cap L = 0.5 and checks
the contraction theorem T(W) = (1 - alpha) W + alpha U(W): predicted
rho = (1 - alpha) + alpha * L vs the empirically measured contraction.
"""

import numpy as np

from ghosh_research.dynamo import (
    build_update_operator,
    predicted_rho,
    empirical_contraction,
    convergence_run,
)

rng = np.random.default_rng(0)
ALPHA, L, N = 0.3, 0.5, 32

P = build_update_operator(rng, N, L, d=3)
print(f"spectral norm of U: {np.linalg.svd(P, compute_uv=False)[0]:.4f} (cap L = {L})")

rho_pred = predicted_rho(ALPHA, L)
emp_mean, emp_max = empirical_contraction(rng, P, ALPHA, n_pairs=4, n_steps=50, n=N)
print(f"predicted rho = {rho_pred:.4f}; empirical mean = {emp_mean:.4f}, max = {emp_max:.4f}")

traj, converged = convergence_run(rng, P, ALPHA, n_steps=50, n=N)
print(f"converged within 50 steps: {converged} (final relative norm {traj[-1]:.2e})")
print("Theorem holds: empirical contraction is below the predicted rho." if emp_max <= rho_pred + 1e-9
      else "WARNING: empirical contraction exceeded the prediction!")
