"""memento — continual learning and the retention/plasticity trade-off.

Original implementation of the core idea of "Multi-Agent Memento" (under
journal revision, 2026): a sliding-window logistic-regression refit (the
fine-tune analogue) versus a never-deleted prototype bank (the no-forgetting
analogue) on a synthetic stream with rotating class boundaries.
"""

from .memento import (
    generate_stream,
    sgd_logistic_fit,
    sliding_window_refit,
    logistic_predict,
    prototype_bank_fit,
    prototype_predict,
    segment_accuracies,
)

__all__ = [
    "generate_stream",
    "sgd_logistic_fit",
    "sliding_window_refit",
    "logistic_predict",
    "prototype_bank_fit",
    "prototype_predict",
    "segment_accuracies",
]
