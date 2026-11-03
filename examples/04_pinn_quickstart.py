#!/usr/bin/env python3
"""Quickstart: hard-negative stress test for LLPS protein classification.

Trains a numpy logistic-regression classifier on a small synthetic protein
panel and prints AUC under the easy/hard negative protocols, plus the
Flory-Huggins physics-loss demo terms. Runs in well under 60 seconds.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ghosh_research import pinn

panel = pinn.make_protein_panel(n_pos=150, n_easy=150, n_hard=150, seed=11)
X, y, kind = panel["X"], panel["y"], panel["kind"]
pos = np.flatnonzero(kind == "pos")
easy = np.flatnonzero(kind == "easy")
hard = np.flatnonzero(kind == "hard")

for name, idx in [("P1: pos vs easy", np.concatenate([pos, easy])),
                  ("P3: pos vs hard", np.concatenate([pos, hard]))]:
    auc = pinn.protocol_auc(X[idx], y[idx], k=3, seed=1)["auc"]
    print(f"{name}: AUC = {auc:.4f}")

phi = np.linspace(0.05, 0.95, 20)
loss = pinn.fhp_loss_demo(phi, (phi > 0.5).astype(int),
                          np.clip((phi > 0.5).astype(float), 0.01, 0.99))
print(f"FHP loss demo total = {loss['total']:.4f} "
      f"(BCE={loss['bce']:.4f}, physics_grad={loss['physics_grad_term']:.4f}, "
      f"physics_perc={loss['physics_perc_term']:.4f})")
