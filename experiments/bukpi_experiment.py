"""Experiment: type-aware correlate() facade on a synthetic enterprise table.

n = 500 rows with planted associations, one per dispatch branch:
  * numeric-numeric:        Pearson r, planted ~0.70
  * numeric-categorical:    ANOVA eta^2, planted ~0.50
  * categorical-categorical: Cramer's V, planted ~0.60
  * text vs binary/ternary sentiment label: hash sketch + k-means (k=3)
    then Cramer's V, planted ~0.95
Plus a robustness check: 100 independent null numeric pairs
(expect mean|r| ~= 0.03, none above 0.2).

Writes results/bukpi/results.md and results/bukpi/charts.png.
Fixed seed for reproducibility.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ghosh_research.bukpi import correlate, pearson_r

SEED = 20261019
N = 500
OUT_DIR = os.path.join("results", "bukpi")

POS = ["excellent", "outstanding", "superb", "remarkable", "exceptional", "fantastic"]
NEG = ["terrible", "awful", "dreadful", "poor", "disappointing", "horrible"]
NEU = ["adequate", "standard", "typical", "ordinary", "average", "common"]
NOUNS = ["product", "service", "delivery", "support", "quality", "response"]


def make_text(rng: np.random.Generator, sentiment: int) -> str:
    # Sentiment-dense review snippet: the planted signal lives in the adjectives.
    adjs = [POS, NEG, NEU][sentiment]
    a1, a2, a3 = rng.choice(adjs, size=3)
    return f"product review: {a1}, {a2}, {a3}."


def main() -> None:
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- planted table ----
    num_a = rng.standard_normal(N)
    num_b = 0.70 * num_a + np.sqrt(1 - 0.70 ** 2) * rng.standard_normal(N)  # r ~= 0.70

    cat_d = rng.choice(["low", "mid", "high"], size=N)
    means = {"low": -np.sqrt(1.5), "mid": 0.0, "high": np.sqrt(1.5)}
    num_c = np.array([means[c] for c in cat_d]) + rng.standard_normal(N)  # eta^2 ~= 0.50

    cat_e = rng.choice(["A", "B", "C"], size=N)
    cat_f = np.array([
        c if rng.random() < 0.72 else rng.choice(["A", "B", "C"])
        for c in cat_e
    ])  # strong diagonal association, V ~= 0.6

    sentiment = rng.choice([0, 1, 2], size=N)  # 0 pos, 1 neg, 2 neutral
    texts = [make_text(rng, int(s)) for s in sentiment]

    r_rng = np.random.default_rng(SEED + 1)
    planted = [
        ("numeric-numeric (Pearson r)", 0.70, num_a, num_b),
        ("numeric-categorical (ANOVA eta^2)", 0.50, num_c, cat_d),
        ("categorical-categorical (Cramer's V)", 0.60, cat_e, cat_f),
    ]
    rows = []
    for branch, planted_v, x, y in planted:
        res = correlate(x, y)
        rows.append((branch, planted_v, res["measure"], res["value"]))

    res_text = correlate(texts, sentiment, rng=r_rng)
    rows.append(("text vs sentiment (cluster Cramer's V)", 0.95,
                 res_text["measure"], res_text["value"]))

    # ---- null-pair robustness: 100 independent numeric pairs ----
    null_rs = []
    for _ in range(100):
        u = rng.standard_normal(N)
        v = rng.standard_normal(N)
        null_rs.append(abs(pearson_r(u, v)))
    null_rs = np.array(null_rs)
    null_mean, null_max = float(null_rs.mean()), float(null_rs.max())

    lines = [
        "# BUKPI — type-aware correlate() facade",
        "",
        f"Seed: {SEED}; n = {N} rows; text branch uses 64-d hash sketches + Lloyd k-means (k=3).",
        "",
        "| branch | planted strength | measure | recovered | status |",
        "|--------|------------------|---------|-----------|--------|",
    ]
    ok_all = True
    for branch, planted_v, measure, value in rows:
        ok = abs(value - planted_v) <= 0.15
        ok_all &= ok
        lines.append(
            f"| {branch} | ~{planted_v:.2f} | {measure} | {value:.3f} | "
            f"{'recovered' if ok else 'MISSED'} |"
        )
    lines += [
        "",
        "## Null-pair robustness (100 independent numeric pairs)",
        "",
        f"- mean |r| = **{null_mean:.4f}** (expect ~= 0.03)",
        f"- max |r| = **{null_max:.4f}** (expect < 0.2): "
        f"**{'PASS' if null_max < 0.2 else 'FAIL'}**",
        "",
        f"All planted associations recovered: **{ok_all}**.",
    ]
    with open(os.path.join(OUT_DIR, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---- charts.png ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = ["num-num\nr", "num-cat\neta^2", "cat-cat\nV", "text\nV"]
    planted_vals = [r[1] for r in rows]
    rec_vals = [r[3] for r in rows]
    xs = np.arange(len(labels))
    axes[0].bar(xs - 0.2, planted_vals, 0.4, label="planted")
    axes[0].bar(xs + 0.2, rec_vals, 0.4, label="recovered")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("association strength")
    axes[0].set_title("Planted vs recovered association per branch")
    axes[0].legend()

    axes[1].hist(null_rs, bins=20, edgecolor="k", alpha=0.7)
    axes[1].axvline(0.2, color="r", linestyle="--", label="0.2 threshold")
    axes[1].set_xlabel("|r| over 100 null pairs")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"Null robustness: mean|r|={null_mean:.3f}, max|r|={null_max:.3f}")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "charts.png"), dpi=120)
    plt.close(fig)

    for branch, planted_v, measure, value in rows:
        print(f"{branch:38s} planted~{planted_v:.2f} recovered={value:.3f}")
    print(f"null pairs: mean|r|={null_mean:.4f} max|r|={null_max:.4f}")


if __name__ == "__main__":
    main()
