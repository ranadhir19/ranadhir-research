"""dynamo — contractive world-model operator verification.

Original implementation of the core idea of "DYNAMO: Dynamic Adaptive
Neural Architecture" (under journal revision, 2026): verifies the
corrected contraction theorem ``T(W) = (1 - alpha) W + alpha U(W)`` with
``U`` spectrally clipped to Lipschitz constant ``L < 1``, comparing the
predicted contraction factor ``rho = (1 - alpha) + alpha L`` against the
empirically measured one.
"""

from .dynamo import (
    spectrally_clipped_matrix,
    build_update_operator,
    contractive_step,
    predicted_rho,
    empirical_contraction,
    convergence_run,
)

__all__ = [
    "spectrally_clipped_matrix",
    "build_update_operator",
    "contractive_step",
    "predicted_rho",
    "empirical_contraction",
    "convergence_run",
]
