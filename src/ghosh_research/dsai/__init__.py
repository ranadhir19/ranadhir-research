"""Theory-augmented option pricing: Black-Scholes + ML residual (DSAI).

Original implementation of the core computational idea in
"Theoretical Augmentation Bridging Analytic Finance and Artificial
Intelligence for Robust Option Pricing in Extreme Market Regimes"
(paper under journal revision, 2026).
"""

from .dsai import (
    FEATURE_NAMES,
    bs_call,
    make_option_panel,
    fit_ridge,
    predict_ridge,
    rmse,
    mae,
    run_pricing_comparison,
)

__all__ = [
    "FEATURE_NAMES",
    "bs_call",
    "make_option_panel",
    "fit_ridge",
    "predict_ridge",
    "rmse",
    "mae",
    "run_pricing_comparison",
]
