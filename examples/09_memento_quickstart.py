"""Quickstart: Memento retention/plasticity trade-off (runs in seconds).

Sliding-window refit vs never-deleted prototype bank on the rotating-boundary
stream; prints per-segment accuracy for both.
"""

import numpy as np

from ghosh_research.memento import (
    generate_stream,
    sliding_window_refit,
    logistic_predict,
    prototype_bank_fit,
    prototype_predict,
    segment_accuracies,
)

rng = np.random.default_rng(3)
X, y = generate_stream(rng)

w = sliding_window_refit(X, y, rng)
bank = prototype_bank_fit(X, y)

for name, pred in [("refit (plastic) ", logistic_predict(X, w)),
                   ("memory (retains)", prototype_predict(X, bank))]:
    acc = segment_accuracies(pred, y)
    print(f"{name}: overall={acc['overall']:.3f} | 0-700={acc['0-700']:.3f} | "
          f"700-1400={acc['700-1400']:.3f} | 1400-2000={acc['1400-2000']:.3f}")
print("Trade-off: memory keeps early segments, refit owns the latest one.")
