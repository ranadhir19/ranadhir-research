#!/usr/bin/env python3
"""PINN experiment: hard-negative stress test + Flory-Huggins loss demo.

Builds a synthetic protein panel (4 biophysical features) with positives,
easy negatives, and hard negatives overlapping the positive region; runs a
numpy logistic-regression classifier under stratified 5-fold CV for three
protocols (P1/P2/P3); and evaluates the paper's Flory-Huggins
physics-informed loss spec on sample composition points. Writes
results/pinn/results.md and results/pinn/charts.png. All randomness seeded.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ghosh_research import pinn  # noqa: E402

SEED = 0
OUT = Path(__file__).resolve().parents[1] / "results" / "pinn"


def main() -> dict:
    panel = pinn.make_protein_panel(n_pos=400, n_easy=400, n_hard=400, seed=SEED)
    X, y, kind = panel["X"], panel["y"], panel["kind"]
    rng = np.random.default_rng(SEED)

    pos_i = np.flatnonzero(kind == "pos")
    easy_i = np.flatnonzero(kind == "easy")
    hard_i = np.flatnonzero(kind == "hard")

    protocols = {
        "P1: positives vs easy negatives": np.concatenate([pos_i, easy_i]),
        "P2: positives vs balanced easy+hard": np.concatenate([
            pos_i,
            rng.choice(easy_i, 200, replace=False),
            rng.choice(hard_i, 200, replace=False),
        ]),
        "P3: positives vs hard negatives": np.concatenate([pos_i, hard_i]),
    }
    results = {}
    for name, idx in protocols.items():
        r = pinn.protocol_auc(X[idx], y[idx], k=5, seed=1)
        results[name] = r["auc"]
        print(f"{name}: AUC = {r['auc']:.4f} (n={len(idx)})")

    # Flory-Huggins physics-informed loss demo on sample composition points
    phi = np.linspace(0.05, 0.95, 50)
    y_demo = (phi > 0.5).astype(int)
    rng_demo = np.random.default_rng(7)
    p_demo = np.clip(y_demo + rng_demo.normal(0, 0.2, size=phi.shape), 0.01, 0.99)
    loss = pinn.fhp_loss_demo(phi, y_demo, p_demo, chi=2.5,
                              lam1=0.1, lam2=0.1, p_perc=0.42, p_c=0.30)
    print("FHP loss demo: " + ", ".join(
        f"{k}={v:.4f}" for k, v in loss.items()
        if isinstance(v, float)))

    OUT.mkdir(parents=True, exist_ok=True)
    md = f"""# PINN experiment results

Config: 400 positives, 400 easy negatives, 400 hard negatives, seed={SEED};
numpy logistic regression, stratified 5-fold CV, pooled out-of-fold AUC.

| protocol | AUC |
|---|---|
| P1: positives vs easy negatives | {results['P1: positives vs easy negatives']:.4f} |
| P2: positives vs balanced easy+hard | {results['P2: positives vs balanced easy+hard']:.4f} |
| P3: positives vs hard negatives | {results['P3: positives vs hard negatives']:.4f} |

Headline accuracy (P1) collapses under the hard-negative stress test (P3).

## Flory-Huggins physics-informed loss demo

Loss spec L = BCE + lam1*|dG_FH/dphi|^2 + lam2*(p_perc - p_c)^2 evaluated on
50 sample composition points (chi=2.5, lam1=lam2=0.1, p_perc=0.42, p_c=0.30):

| term | value |
|---|---|
| BCE | {loss['bce']:.4f} |
| physics gradient term | {loss['physics_grad_term']:.4f} |
| physics percolation term | {loss['physics_perc_term']:.4f} |
| total | {loss['total']:.4f} |

Note: this demonstrates the loss is computable; it is not a trained network.
"""
    (OUT / "results.md").write_text(md)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = ["P1\npos vs easy", "P2\npos vs easy+hard", "P3\npos vs hard"]
    aucs = [results[k] for k in protocols]
    bars = ax.bar(names, aucs)
    ax.set_ylabel("ROC AUC (5-fold pooled)")
    ax.set_title("Hard-negative stress test: AUC vs negative difficulty")
    ax.set_ylim(0.5, 1.02)
    for b, v in zip(bars, aucs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(OUT / "charts.png", dpi=120)
    plt.close(fig)
    print(f"wrote {OUT / 'results.md'} and {OUT / 'charts.png'}")
    return {"protocols": results, "fhp_loss": loss}


if __name__ == "__main__":
    main()
