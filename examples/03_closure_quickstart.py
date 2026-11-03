#!/usr/bin/env python3
"""Quickstart: hybrid-discovery simulation (Agent Closure Theorem).

Simulates a pure-AI agent vs a hybrid agent under the paper's two-regime
setup and prints measured novel-discovery rates against theory.
Runs in well under 60 seconds.

Note: the paper is purely theoretical; this is its canonical simulation,
not a reproduction of empirical results.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ghosh_research import closure

res = closure.simulate(n_steps=600, n_seeds=8, p_novel=0.30, alpha=0.05,
                       base_seed=3)
pure, hybrid, theory = res["pure"], res["hybrid"], res["theory"]
print(f"pure AI agent: novel rate = {pure['mean']:.4f} (theory: {theory['pure']:.4f})")
print(f"hybrid agent:  novel rate = {hybrid['mean']:.4f} (theory: {theory['hybrid']:.4f})")
print("The pure agent cannot exceed the alpha=0.05 closure bound; "
      "the hybrid agent breaks it via novel-regime observations.")
