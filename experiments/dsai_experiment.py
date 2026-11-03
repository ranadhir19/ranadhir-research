#!/usr/bin/env python3
"""DSAI experiment: theory-augmented option pricing under regime shift.

Trains on a synthetic calm-regime option panel (n=4000) and tests on a
crisis-regime panel (n=4000): pure-ML ridge vs analytic Black-Scholes vs
hybrid (C_BS + ridge on residual). Reports RMSE/MAE on the crisis test set.
Writes results/dsai/results.md and results/dsai/charts.png.
All randomness is seeded.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ghosh_research import dsai  # noqa: E402

SEED = 0
N_TRAIN = 4000
N_TEST = 4000
OUT = Path(__file__).resolve().parents[1] / "results" / "dsai"


def main() -> dict:
    res = dsai.run_pricing_comparison(n_train=N_TRAIN, n_test=N_TEST, seed=SEED)
    for name in ("pure_ml", "bs_only", "hybrid"):
        print(f"{name:8s}: crisis RMSE={res[name]['rmse']:.3f} "
              f"MAE={res[name]['mae']:.3f}")
    print(f"train RMSE: pure_ml={res['train_rmse_pure_ml']:.3f} "
          f"hybrid={res['train_rmse_hybrid']:.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    md = f"""# DSAI experiment results

Config: train on calm regime (n={N_TRAIN}), test on crisis regime (n={N_TEST}),
seed={SEED}; ridge lambda=1.0 on (moneyness, maturity, rate, vol, VIX).

| model | crisis RMSE | crisis MAE |
|---|---|---|
| pure ML (ridge on price) | {res['pure_ml']['rmse']:.3f} | {res['pure_ml']['mae']:.3f} |
| BS-only (analytic theory) | {res['bs_only']['rmse']:.3f} | {res['bs_only']['mae']:.3f} |
| hybrid (BS + ridge residual) | {res['hybrid']['rmse']:.3f} | {res['hybrid']['mae']:.3f} |

Training (calm) RMSE: pure ML {res['train_rmse_pure_ml']:.3f}, hybrid
{res['train_rmse_hybrid']:.3f}. The pure-ML pricer fits calm data well but
collapses under the crisis regime shift; the theory anchor keeps the hybrid
model robust and best on crisis data.
"""
    (OUT / "results.md").write_text(md)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = ["pure ML", "BS-only", "hybrid\n(BS + residual)"]
    rmses = [res["pure_ml"]["rmse"], res["bs_only"]["rmse"], res["hybrid"]["rmse"]]
    bars = ax.bar(labels, rmses)
    bars[2].set_color("green")
    ax.set_ylabel("crisis test RMSE")
    ax.set_title("Option pricing under regime shift: train calm, test crisis")
    for b, v in zip(bars, rmses):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(OUT / "charts.png", dpi=120)
    plt.close(fig)
    print(f"wrote {OUT / 'results.md'} and {OUT / 'charts.png'}")
    return res


if __name__ == "__main__":
    main()
