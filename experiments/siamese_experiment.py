"""Experiment: contrastive CAR/LAR verification.

Synthetic cheque pairs (CAR like "$125.00", LAR like "one hundred twenty
five dollars"); negatives are same-magnitude mismatched amounts.  Bigram-hash
character features -> |f_car - f_lar| pair vectors.  Compares on 1500 test
pairs, over 5 seeds:

  * raw-feature-distance baseline (threshold tuned on train),
  * purely linear classifier (logistic regression) on the pair vectors,
  * 16-d linear projection trained with contrastive loss (margin 1.0).

Writes results/siamese/results.md and results/siamese/charts.png.
Fixed seeds for reproducibility.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ghosh_research.siamese import (
    generate_pairs,
    train_projection,
    embed_distances,
    accuracy_at_threshold,
    best_threshold,
    train_linear_classifier,
    linear_predict,
)

SEED = 20261019
N_TRAIN = 4000
N_TEST = 1500
N_SEEDS = 5
OUT_DIR = os.path.join("results", "siamese")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for s in range(N_SEEDS):
        rng = np.random.default_rng(SEED + s)
        Xtr, ytr = generate_pairs(rng, N_TRAIN)
        Xte, yte = generate_pairs(rng, N_TEST)

        raw_tr = np.linalg.norm(Xtr, axis=1)
        thr_raw = best_threshold(raw_tr, ytr)
        acc_raw = accuracy_at_threshold(np.linalg.norm(Xte, axis=1), yte, thr_raw)

        w, b = train_linear_classifier(Xtr, ytr, rng)
        acc_lin = float((linear_predict(Xte, w, b) == yte).mean())

        W = train_projection(Xtr, ytr, rng)
        thr_emb = best_threshold(embed_distances(Xtr, W), ytr)
        acc_sia = accuracy_at_threshold(embed_distances(Xte, W), yte, thr_emb)

        rows.append({"seed": s, "baseline": acc_raw, "linear": acc_lin, "siamese": acc_sia})
        print(f"seed {s}: baseline={acc_raw:.4f} linear={acc_lin:.4f} siamese={acc_sia:.4f}")

    def mean_std(key: str) -> tuple[float, float]:
        v = np.array([r[key] for r in rows])
        return float(v.mean()), float(v.std(ddof=1))

    mb, sb = mean_std("baseline")
    ml, sl = mean_std("linear")
    ms, ss = mean_std("siamese")

    lines = [
        "# SIAMESE — contrastive CAR/LAR verification",
        "",
        f"Seeds: {SEED}..{SEED + N_SEEDS - 1}; {N_TRAIN} train / {N_TEST} test pairs per seed; "
        "16-d linear projection, contrastive margin 1.0, 30 SGD epochs; hard same-magnitude negatives.",
        "",
        "| seed | raw-distance baseline | linear classifier | siamese embedding |",
        "|------|-----------------------|-------------------|-------------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['seed']} | {r['baseline']:.4f} | {r['linear']:.4f} | {r['siamese']:.4f} |"
        )
    lines += [
        f"| **mean +/- std** | **{mb:.4f} +/- {sb:.4f}** | **{ml:.4f} +/- {sl:.4f}** | **{ms:.4f} +/- {ss:.4f}** |",
        "",
        "Reading: the raw-distance baseline (0.50) and the purely linear classifier (0.52) sit at "
        "chance, reproducing the paper's 'linear similarity is insufficient' intuition.  The siamese "
        "embedding reaches 0.98 on clean synthetic pairs: the 'linear projection + norm' objective is "
        "secretly a rank-16 quadratic model (||Wd||^2 = d'WW'd), which discovers compositional "
        "digit<->word bigram correspondences.  The paper's ~54% for the embedding presumably reflects "
        "real-world cheque noise (OCR errors, handwriting, abbreviations) absent from this noiseless "
        "synthetic benchmark -- a deviation from the brief's expectation, reported here as measured.",
    ]
    with open(os.path.join(OUT_DIR, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---- charts.png ----
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(N_SEEDS)
    width = 0.25
    ax.bar(xs - width, [r["baseline"] for r in rows], width, label="raw-distance baseline")
    ax.bar(xs, [r["linear"] for r in rows], width, label="linear classifier")
    ax.bar(xs + width, [r["siamese"] for r in rows], width, label="siamese embedding")
    for m, lab, x in [(mb, "baseline", -width), (ml, "linear", 0.0), (ms, "siamese", width)]:
        ax.hlines(m, xs[0] + x - width / 2, xs[-1] + x + width / 2,
                  colors="k", linestyles="--", alpha=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"seed {r['seed']}" for r in rows])
    ax.set_ylabel("test accuracy (1500 pairs)")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("CAR/LAR pair-verification accuracy (dashed = mean)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "charts.png"), dpi=120)
    plt.close(fig)

    print(f"\nmeans: baseline={mb:.4f}+/-{sb:.4f} linear={ml:.4f}+/-{sl:.4f} siamese={ms:.4f}+/-{ss:.4f}")


if __name__ == "__main__":
    main()
