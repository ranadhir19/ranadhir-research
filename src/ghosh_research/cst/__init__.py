"""Coarse Set Theory for ethical AI: greedy maximal ethical subset construction.

Original implementation of the core computational idea in
"Coarse Set Theory for Ethical AI: A Mathematical Framework for
Multi-Constraint Decision Making with Formal Guarantees"
(paper under journal revision, 2026).
"""

from .cst import (
    sigmoid,
    make_synthetic_lending,
    tune_threshold,
    demographic_parity_gap,
    equalized_opportunity_gap,
    accuracy,
    greedy_ethical_subset,
    subset_report,
)

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
