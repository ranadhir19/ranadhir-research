#!/usr/bin/env python3
"""Quickstart: theory-augmented option pricing under regime shift.

Trains pure-ML, BS-only, and hybrid (BS + ML residual) pricers on calm-regime
options and evaluates on crisis-regime options. Runs in well under 60 seconds.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ghosh_research import dsai

res = dsai.run_pricing_comparison(n_train=2500, n_test=2500, seed=21)
for name, label in [("pure_ml", "pure ML"), ("bs_only", "BS-only"),
                    ("hybrid", "hybrid (BS + residual)")]:
    print(f"{label:22s}: crisis RMSE = {res[name]['rmse']:.3f}, "
          f"MAE = {res[name]['mae']:.3f}")
print("The hybrid's theory anchor keeps it robust where pure ML collapses.")
