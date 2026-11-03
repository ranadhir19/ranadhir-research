"""Quickstart: Bayesian concept-map edge validation (runs in seconds).

Validates one real edge (y = 1.5 x + noise) and one null edge (beta* = 0)
with the Metropolis-Hastings sampler + Savage-Dickey Bayes factor.
"""

import numpy as np

from ghosh_research.kpi import validate_edge

rng = np.random.default_rng(1)
n = 200

# a genuine edge
x = rng.normal(0, 1.5, n)
res = validate_edge(x, 1.5 * x + rng.normal(0, 0.8, n), rng, n_iter=4000, burn=1000)
print(f"true edge:  beta_hat = {res['beta_hat']:.3f} (truth 1.500), "
      f"95% HDI = [{res['hdi_lo']:.3f}, {res['hdi_hi']:.3f}], BF10 = {res['bf10']:.2g}")

# a null edge
x0 = rng.normal(0, 1.0, n)
res0 = validate_edge(x0, rng.normal(0, 1.0, n), rng, n_iter=4000, burn=1000)
print(f"null edge:  beta_hat = {res0['beta_hat']:.3f} (truth 0.000), "
      f"95% HDI = [{res0['hdi_lo']:.3f}, {res0['hdi_hi']:.3f}], BF01 = {res0['bf01']:.1f}")
print("Null edge correctly rejected" if res0["bf01"] > 10 else "Null edge NOT rejected")
