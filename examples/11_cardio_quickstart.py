#!/usr/bin/env python3
"""Quickstart: horizon durability of ECG risk (CARDIO).

Fits a Cox model on a small synthetic cohort (n=800), prints log-HRs and
Grambsch-Therneau p-values, one interaction-information value, and one
knowledge-graph tension. Runs in well under 60 seconds.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ghosh_research import cardio as C

t0 = time.time()
d = C.simulate_cohort(n=800, seed=22, het_k=2.0)
X, Y, delta = d["X"], d["Y"], d["delta"]

fit = C.fit_cox(X, Y, delta)
print("log-HRs:", dict(zip(d["names"], [round(float(b), 3)
                                        for b in fit["beta"]])))

gt = C.grambsch_therneau(fit["beta"], X, Y, delta)
print("GT p-values (time variation):",
      dict(zip(d["names"], [round(float(p), 3) for p in gt["p_values"]])))
print("all p > 0.05:", bool((gt["p_values"] > 0.05).all()))

b = d["blocks"]
ii = C.interaction_information(b["demo"], b["ecg"], Y, delta)
print(f"II(demo;ecg) = {ii['II']:+.1f} nats (negative = redundant views)")

kg = C.ClaimGraph()
kg.add_claim("theorem", "C_tau", "invariant", axis="tau",
             statement="Untruncated concordance cannot depend on the horizon.")
kg.add_claim("measured", "C_tau", "trend_down", axis="tau",
             statement="Incremental concordance falls with the horizon.")
tensions = kg.detect_tensions()
print(f"knowledge-graph tensions detected: {len(tensions)}")
for inv, tr in tensions:
    print(f"  {inv['variable']}: {inv['kind']}/{inv['relation']} vs "
          f"{tr['kind']}/{tr['relation']}")

print(f"done in {time.time() - t0:.1f}s")
