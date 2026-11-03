# KPI — Bayesian Validation of Concept-Map Edges

## The paper

"Bayesian Validation of Concept-Map Edges for KPI Governance" (under journal
revision, 2026).

A concept map asserts causal links between KPIs ("ad spend drives revenue"), and
governance needs a principled way to keep, question, or drop each link as data
accumulates. The paper treats every edge as a Bayesian linear regression
`y = beta * x + eps` and validates it two ways: the posterior of `beta` must
concentrate near the asserted strength, and a Savage-Dickey Bayes factor must
favor the edge's existence over the null of no relationship.

## Core contribution implemented

An original implementation of the paper's validation machinery (no paper text reused):

- Random-walk Metropolis-Hastings sampler for `(beta, log sigma)` in numpy,
  Laplace-informed start (OLS init, proposal scales from the OLS standard error).
- Savage-Dickey `BF10 = N(0; 0, sigma_beta) / N(0; beta_hat, sd(beta_hat))`,
  computed in log space so overwhelming evidence saturates at `inf` cleanly.
- Central 95% credible intervals ("HDI") from posterior samples.

Run: `PYTHONPATH=src python3 experiments/kpi_experiment.py` (seed `20261019`;
13 edges x 300 points; 25000 MH iterations with 8000 burn-in per edge;
prior `beta ~ N(0, 10^2)`). Takes about 2 minutes.

## Results

Measured numbers (from `results/kpi/results.md`):

- Edges recovered within 9% relative error: **9/12**. The three misses are the
  weakest signals — `uptime ~ load` (13.3%), `csat ~ wait_min` (11.8%), and
  `output ~ overtime` (33.0%, slope 0.02 against noise 0.5) — exactly where a
  Bayesian procedure should be uncertain.
- Central-95% HDI coverage of the true slope: **92%** (12/13).
- Null edge (`beta* = 0`): `BF01 = 53.5`, strong evidence *for* the null — the
  absent edge is correctly rejected even though the posterior mean happened to
  draw 0.0914 on this seed.

## Insights

- The sampler is honest about weak edges: point estimates miss the 9% bar on
  low-signal links, but their HDIs still cover the truth, so the interval — not
  the point estimate — is the reliable governance signal.
- Savage-Dickey factors behave as theory predicts: astronomically large `BF10`
  for real edges (up to `>1e300`), and `BF01 = 53.5` for the null edge, cleanly
  separating "link exists" from "no link" on one scale.
- MH acceptance rates held at 0.22–0.24 across all edges, indicating the
  Laplace-informed proposal scales were well tuned with no per-edge fiddling.
