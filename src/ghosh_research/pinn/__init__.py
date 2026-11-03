"""Physics-informed LLPS protein classification + hard-negative stress test.

Original implementation of the core computational ideas in
"Physics-Informed Neural Networks for Predicting Liquid-Liquid Phase
Separation in Neurodegenerative Disease Proteins"
(paper under journal revision, 2026).
"""

from .pinn import (
    FEATURE_NAMES,
    make_protein_panel,
    fit_logreg,
    predict_proba,
    auc_score,
    stratified_kfold,
    protocol_auc,
    flory_huggins,
    d_flory_huggins,
    fhp_loss_demo,
)

__all__ = [
    "FEATURE_NAMES",
    "make_protein_panel",
    "fit_logreg",
    "predict_proba",
    "auc_score",
    "stratified_kfold",
    "protocol_auc",
    "flory_huggins",
    "d_flory_huggins",
    "fhp_loss_demo",
]
