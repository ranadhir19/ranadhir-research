#!/usr/bin/env python3
"""Closure experiment: hybrid-discovery simulation (Agent Closure Theorem).

Simulates a pure-AI agent (regime 0 only) and a hybrid agent (observes the
novel regime 1 with p=0.30/step) for 2000 steps x 30 seeds, applying the
|z| > 1.96 novelty test under the N(0,1) prior. Compares measured novel-flag
rates against the paper's theoretical bound. Writes
results/closure/results.md and results/closure/charts.png.
All randomness is seeded.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ghosh_research import closure  # noqa: E402

N_STEPS = 2000
N_SEEDS = 30
P_NOVEL = 0.30
ALPHA = 0.05
OUT = Path(__file__).resolve().parents[1] / "results" / "closure"


def main() -> dict:
    res = closure.simulate(n_steps=N_STEPS, n_seeds=N_SEEDS,
                           p_novel=P_NOVEL, alpha=ALPHA, base_seed=0)
    pure, hybrid, theory = res["pure"], res["hybrid"], res["theory"]
    print(f"pure agent:   novel rate = {pure['mean']:.4f} +- {pure['ci95']:.4f} "
          f"(theory: {theory['pure']:.4f})")
    print(f"hybrid agent: novel rate = {hybrid['mean']:.4f} +- {hybrid['ci95']:.4f} "
          f"(theory: {theory['hybrid']:.4f})")

    OUT.mkdir(parents=True, exist_ok=True)
    md = f"""# Closure experiment results

Config: {N_STEPS} steps x {N_SEEDS} seeds, p_novel={P_NOVEL}, alpha={ALPHA},
novelty test |z| > {theory['z_crit']:.4f} under the N(0,1) prior.

| agent | measured novel rate | 95% CI | theoretical |
|---|---|---|---|
| pure AI (regime 0 only) | {pure['mean']:.4f} | +- {pure['ci95']:.4f} | {theory['pure']:.4f} |
| hybrid (30% novel-regime obs.) | {hybrid['mean']:.4f} | +- {hybrid['ci95']:.4f} | {theory['hybrid']:.4f} |

The pure agent's novel rate sits at the significance level alpha: confined to
its prior, it cannot transcend it. The hybrid agent's rate matches the
mixture prediction p * power + (1-p) * alpha with detection power
{theory['detection_power']:.4f} on the N(4,1) novel regime.
"""
    (OUT / "results.md").write_text(md)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["pure AI agent", "hybrid agent"]
    means = [pure["mean"], hybrid["mean"]]
    cis = [pure["ci95"], hybrid["ci95"]]
    theory_vals = [theory["pure"], theory["hybrid"]]
    x = np.arange(2)
    ax.bar(x, means, yerr=cis, capsize=8, label="measured +- 95% CI")
    ax.scatter(x, theory_vals, color="red", s=80, zorder=5, label="theory")
    ax.axhline(ALPHA, color="gray", linestyle="--", label="alpha = 0.05 bound")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("novel-flag rate")
    ax.set_title("Agent Closure Theorem: pure vs hybrid discovery rates")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "charts.png", dpi=120)
    plt.close(fig)
    print(f"wrote {OUT / 'results.md'} and {OUT / 'charts.png'}")
    return res


if __name__ == "__main__":
    main()
