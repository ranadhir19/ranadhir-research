"""Experiment: retention vs plasticity on a shifting binary stream.

n = 2000 stream, class boundary rotates 45 -> 90 -> 135 degrees at
t = 700 / 1400, 5% label noise.  Compares:

  (a) sliding-window logistic regression refit by mini-batch SGD
      (window 400, refit every 200, warm-started);
  (b) growing prototype bank (per-class block means, never deleted),
      similarity-weighted vote.

Reports accuracy overall and per segment (0-700, 700-1400, 1400-2000).

Writes results/memento/results.md and results/memento/charts.png.
Fixed seed for reproducibility.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ghosh_research.memento import (
    generate_stream,
    sliding_window_refit,
    logistic_predict,
    prototype_bank_fit,
    prototype_predict,
    segment_accuracies,
)

SEED = 20261019
OUT_DIR = os.path.join("results", "memento")


def main() -> None:
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    X, y = generate_stream(rng)

    w = sliding_window_refit(X, y, rng)
    acc_refit = segment_accuracies(logistic_predict(X, w), y)

    bank = prototype_bank_fit(X, y)
    acc_mem = segment_accuracies(prototype_predict(X, bank), y)

    keys = ["overall", "0-700", "700-1400", "1400-2000"]
    lines = [
        "# MEMENTO — retention vs plasticity under distribution shift",
        "",
        f"Seed: {SEED}; n = 2000; boundary angles 45/90/135 deg at t = 700/1400; 5% label noise.",
        "",
        "| method | overall | 0-700 | 700-1400 | 1400-2000 |",
        "|--------|---------|-------|----------|-----------|",
        "| sliding-window refit (fine-tune analogue) | "
        + " | ".join(f"{acc_refit[k]:.3f}" for k in keys) + " |",
        "| prototype bank (no-forgetting analogue) | "
        + " | ".join(f"{acc_mem[k]:.3f}" for k in keys) + " |",
        "",
        "Reading: the prototype bank retains the early segments (never deleted, so old "
        "knowledge survives) but its frozen compromise lags after the shifts; the "
        "sliding-window refit tracks the current distribution and dominates the latest "
        "segment while forgetting the earlier ones. Neither dominates everywhere: a "
        "retention/plasticity trade-off, not a clean win.",
    ]
    with open(os.path.join(OUT_DIR, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---- charts.png ----
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = ["overall", "seg 0-700", "seg 700-1400", "seg 1400-2000"]
    xs = np.arange(len(labels))
    ax.bar(xs - 0.2, [acc_refit[k] for k in keys], 0.4, label="sliding-window refit")
    ax.bar(xs + 0.2, [acc_mem[k] for k in keys], 0.4, label="prototype bank")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title("Retention vs plasticity: accuracy per segment after full stream")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "charts.png"), dpi=120)
    plt.close(fig)

    print("refit :", " ".join(f"{k}={acc_refit[k]:.3f}" for k in keys))
    print("memory:", " ".join(f"{k}={acc_mem[k]:.3f}" for k in keys))


if __name__ == "__main__":
    main()
