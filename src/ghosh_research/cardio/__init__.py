"""Cardiovascular risk from the resting ECG: horizon durability, multi-modal
redundancy, and knowledge-graph-guided testing (CARDIO).

Original implementation of the core computational ideas in
"Preliminary investigation: forecasting cardiovascular risk from the resting
ECG" (Dr. Ranadhir Ghosh, shared with Dr. Debabrata Mukhopadhyay, Mayo Clinic,
Sep 2026), rendered on synthetic constant-effect proportional-hazards data.
"""

from .cardio import (
    FEATURE_NAMES,
    simulate_cohort,
    partial_loglik,
    fit_cox,
    schoenfeld,
    grambsch_therneau,
    km_survival,
    km_eval,
    breslow,
    h0_eval,
    predict_surv,
    uno_concordance,
    td_auc,
    ipcw_brier,
    ipa,
    landmark_cohort,
    interaction_information,
    ClaimGraph,
    GatedPipeline,
    fit_gated_fusion,
)

__all__ = [
    "FEATURE_NAMES",
    "simulate_cohort",
    "partial_loglik",
    "fit_cox",
    "schoenfeld",
    "grambsch_therneau",
    "km_survival",
    "km_eval",
    "breslow",
    "h0_eval",
    "predict_surv",
    "uno_concordance",
    "td_auc",
    "ipcw_brier",
    "ipa",
    "landmark_cohort",
    "interaction_information",
    "ClaimGraph",
    "GatedPipeline",
    "fit_gated_fusion",
]
