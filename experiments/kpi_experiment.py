"""Experiment: Bayesian concept-map edge validation benchmark.

Synthetic benchmark of 12 edges with known slopes beta* (varied x-spread
and noise, including a steep low-spread edge) plus one null edge with
beta* = 0.  Each edge is validated with the MH sampler; we report the
fraction of edges recovered within 9% relative error, the central-95%
HDI coverage of beta*, and BF01 for the null edge (expect large).

Writes results/kpi/results.md and results/kpi/charts.png.
Fixed seed for reproducibility.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ghosh_research.kpi import validate_edge

SEED = 20261019
N = 300
OUT_DIR = os.path.join("results", "kpi")

# (name, beta*, x std, noise sigma); last entry is the null edge.
EDGES = [
    ("revenue ~ ad_spend", 0.80, 2.0, 0.50),
    ("churn ~ latency", -1.50, 1.0, 0.80),
    ("margin ~ discount", -22.0, 0.10, 1.00),
    ("nps ~ response_hrs", -0.35, 3.0, 1.20),
    ("uptime ~ load", 0.060, 5.0, 0.40),
    ("conversion ~ adstock", 2.40, 0.8, 1.50),
    ("retention ~ tickets", -0.90, 1.5, 0.70),
    ("cost ~ volume", 1.10, 2.5, 2.00),
    ("csat ~ wait_min", -0.12, 4.0, 0.60),
    ("growth ~ hiring", 0.45, 1.2, 0.70),
    ("risk ~ leverage", 3.30, 0.6, 1.10),
    ("output ~ overtime", 0.020, 8.0, 0.50),
    ("null: noise ~ noise", 0.00, 1.0, 1.00),
]


def main() -> None:
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = []
    for name, beta_star, sx, sigma in EDGES:
        x = rng.normal(0.0, sx, N)
        y = beta_star * x + rng.normal(0.0, sigma, N)
        res = validate_edge(x, y, rng)
        bhat = res["beta_hat"]
        rel_err = abs(bhat - beta_star) / abs(beta_star) if beta_star != 0 else float("nan")
        covers = res["hdi_lo"] <= beta_star <= res["hdi_hi"]
        rows.append({"name": name, "beta_star": beta_star, **res,
                     "rel_err": rel_err, "covers": covers})
        print(f"{name:22s} beta*={beta_star:8.3f} bhat={bhat:8.4f} "
              f"rel_err={rel_err:6.3%} covers={covers} BF01={res['bf01']:.1f} "
              f"acc={res['accept_rate']:.2f}")

    nonzero = [r for r in rows if r["beta_star"] != 0.0]
    within9 = sum(r["rel_err"] <= 0.09 for r in nonzero)
    coverage = sum(r["covers"] for r in rows) / len(rows)
    null_row = rows[-1]

    def fmt_bf(v: float) -> str:
        if np.isfinite(v):
            return f"{v:.3g}"
        return ">1e300" if v > 0 else "<1e-300"

    lines = [
        "# KPI — Bayesian concept-map edge validation",
        "",
        f"Seed: {SEED}; n = {N} per edge; MH 25000 iterations (8000 burn-in); prior beta ~ N(0, 10^2).",
        "",
        "| edge | beta* | beta_hat | rel. err | 95% HDI | covers beta* | BF10 | BF01 |",
        "|------|-------|----------|----------|---------|----------------|------|------|",
    ]
    for r in rows:
        re = f"{r['rel_err']:.2%}" if r["beta_star"] != 0 else "n/a"
        lines.append(
            f"| {r['name']} | {r['beta_star']:.3f} | {r['beta_hat']:.4f} | {re} "
            f"| [{r['hdi_lo']:.4f}, {r['hdi_hi']:.4f}] | {r['covers']} "
            f"| {fmt_bf(r['bf10'])} | {fmt_bf(r['bf01'])} |"
        )
    lines += [
        "",
        f"- Edges recovered within 9% relative error: **{within9}/{len(nonzero)}**.",
        f"- Central-95% HDI coverage of beta*: **{coverage:.0%}** ({sum(r['covers'] for r in rows)}/{len(rows)}).",
        f"- Null edge BF01 = **{null_row['bf01']:.1f}** (strong evidence for the null: the absent edge is correctly rejected).",
    ]
    with open(os.path.join(OUT_DIR, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---- charts.png: forest plot ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ys = np.arange(len(rows))
    for i, r in enumerate(rows):
        ax.errorbar(r["beta_hat"], i,
                    xerr=[[r["beta_hat"] - r["hdi_lo"]], [r["hdi_hi"] - r["beta_hat"]]],
                    fmt="o", capsize=4, color="C0")
        ax.plot(r["beta_star"], i, "rx", ms=9, mew=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["name"] for r in rows], fontsize=8)
    ax.axvline(0.0, color="k", linestyle=":", alpha=0.5)
    ax.set_xlabel("slope beta")
    ax.set_title("Posterior mean +/- 95% HDI (blue) vs true beta* (red x)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "charts.png"), dpi=120)
    plt.close(fig)

    print(f"\nwithin 9%: {within9}/{len(nonzero)}; HDI coverage: {coverage:.0%}; null BF01: {null_row['bf01']:.1f}")


if __name__ == "__main__":
    main()
