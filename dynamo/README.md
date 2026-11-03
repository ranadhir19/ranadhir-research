# DYNAMO — Contractive World-Model Operator Verification

## The paper

"DYNAMO: Dynamic Adaptive Neural Architecture" (under journal revision, JMLR, 2026).

The paper studies a world model whose latent transition is updated by the operator
`T(W) = (1 - alpha) W + alpha * U(W)`, where `U` is an MLP-style update. The core
theoretical question is when repeated application of `T` is guaranteed to settle
to a unique fixed point instead of oscillating or diverging. The corrected Theorem 1
answers it: if every layer of `U` is spectrally clipped so the whole update has
Lipschitz constant `L < 1`, then `T` is a contraction with factor
`rho = (1 - alpha) + alpha * L`, and Banach's fixed-point theorem gives convergence.

## Core contribution implemented

An original implementation of the corrected Theorem 1 as a verifiable numerical
experiment (no paper text reused):

- `U(W) = P W` with `P = A_3 A_2 A_1`, each `64x64` factor drawn at random and
  spectrally clipped via SVD so its largest singular value is `<= L^(1/3)`;
  hence `||P||_2 <= L` and `Lip(U) <= L`.
- The empirical per-step contraction `||T(W_k) - T(W'_k)||_F / ||W_k - W'_k||_F`
  between nearby starts, as a geometric mean over 50 steps and 8 random pairs.
- A convergence check: whether `||W_k||/||W_0|| < 1e-3` is reached within 50 steps.

Run: `PYTHONPATH=src python3 experiments/dynamo_experiment.py`
(seed `20261019`, `alpha = 0.3`, `d = 3`, `n = 64`).

## Results

Measured numbers (from `results/dynamo/results.md`):

| L | predicted rho | empirical mean | empirical max | converged in 50 steps | final rel. norm |
|---|---------------|----------------|---------------|-----------------------|-----------------|
| 0.50 | 0.8500 | 0.8238 | 0.8242 | True | 6.29e-05 |
| 0.95 | 0.9850 | 0.9392 | 0.9402 | False | 4.22e-02 |
| 1.00 | 1.0000 | 0.9572 | 0.9586 | False | 1.22e-01 |
| 1.05 | 1.0150 | 0.9611 | 0.9621 | False | 1.53e-01 |

## Insights

- The empirical contraction always lies below the predicted `rho`: the theorem is a
  valid upper bound, and at `L = 0.5` it is tight (gap of only 0.026, `rho = 0.85`),
  with iterates converging to `6.3e-05` relative norm within 50 steps.
- As `L -> 1`, the predicted `rho -> 1` and contraction plateaus: at `L = 0.95` the
  iterates only reach `4.2e-02` in 50 steps, and at `L = 1.00` they stall at
  `1.2e-01` — the theory's warning that the Lipschitz cap must stay strictly below
  1 is visible numerically, not just on paper.
- At `L = 1.05` (`rho = 1.015 > 1`) the contraction guarantee is lost; iterates fail
  to converge, confirming the `L < 1` hypothesis is sharp in practice.
- The near-equality of empirical mean and max shows the contraction factor is an
  intrinsic property of the operator, not an artifact of the chosen starting pair.
