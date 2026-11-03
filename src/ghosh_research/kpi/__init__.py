"""kpi — Bayesian validation of concept-map edges for KPI governance.

Original implementation of the core idea of "Bayesian Validation of
Concept-Map Edges for KPI Governance" (under journal revision, 2026):
a random-walk Metropolis-Hastings sampler for Bayesian linear regression
on ``(beta, log sigma)`` plus Savage-Dickey Bayes factors for edge
existence.
"""

from .kpi import (
    log_posterior,
    mh_sample,
    savage_dickey_bf10,
    hdi_central,
    validate_edge,
)

__all__ = [
    "log_posterior",
    "mh_sample",
    "savage_dickey_bf10",
    "hdi_central",
    "validate_edge",
]
