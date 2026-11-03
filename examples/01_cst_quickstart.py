#!/usr/bin/env python3
"""Quickstart: greedy maximal ethical subset (Coarse Set Theory).

Runs the paper's greedy construction on a small synthetic lending
population and prints fairness gaps before/after plus the accuracy cost.
Runs in well under 60 seconds.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ghosh_research import cst

data = cst.make_synthetic_lending(n=1000, seed=1)
threshold = cst.tune_threshold(data["scores"], data["labels"])
pred = (data["scores"] >= threshold).astype(int)

print(f"baseline: DP gap = {cst.demographic_parity_gap(pred, data['group']):.4f}, "
      f"EO gap = {cst.equalized_opportunity_gap(pred, data['labels'], data['group']):.4f}, "
      f"accuracy = {cst.accuracy(pred, data['labels']):.4f}")

run = cst.greedy_ethical_subset(
    data["features"], pred, data["scores"], data["labels"], data["group"],
    tau_dp=0.015, tau_eo=0.015, eps=0.0, delta=0.1, seed=1,
)
rep = cst.subset_report(pred, data["labels"], data["group"], run["admitted"])
print(f"admitted {rep['n_admitted']}/{rep['n_total']} decisions: "
      f"DP gap = {rep['dp_gap_admitted']:.4f}, "
      f"EO gap = {rep['eo_gap_admitted']:.4f}, "
      f"accuracy = {rep['accuracy_admitted']:.4f}")
print(f"accuracy cost: {100*(rep['accuracy_full']-rep['accuracy_admitted']):.2f} pp")
