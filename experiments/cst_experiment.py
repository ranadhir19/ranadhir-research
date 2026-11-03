#!/usr/bin/env python3
"""CST experiment: greedy maximal ethical subset on synthetic lending data.

Builds a synthetic lending population (n=2000), trains a globally-tuned
threshold classifier, then runs the Coarse Set Theory greedy construction
to select a maximal subset of decisions satisfying demographic-parity and
equal-opportunity gap bounds. Writes results/cst/results.md and
results/cst/charts.png. All randomness is seeded.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ghosh_research import cst  # noqa: E402

SEED = 0
N = 2000
TAU_DP = 0.015
TAU_EO = 0.015
EPS = 0.0
DELTA = 0.1
OUT = Path(__file__).resolve().parents[1] / "results" / "cst"


def main() -> dict:
    data = cst.make_synthetic_lending(n=N, seed=SEED)
    threshold = cst.tune_threshold(data["scores"], data["labels"])
    pred = (data["scores"] >= threshold).astype(int)

    run = cst.greedy_ethical_subset(
        data["features"], pred, data["scores"], data["labels"], data["group"],
        tau_dp=TAU_DP, tau_eo=TAU_EO, eps=EPS, delta=DELTA, seed=SEED,
    )
    rep = cst.subset_report(pred, data["labels"], data["group"], run["admitted"])
    rep["threshold"] = threshold
    rep["admission_rate"] = run["admission_rate"]
    rep["accuracy_cost_pp"] = 100 * (rep["accuracy_full"] - rep["accuracy_admitted"])

    print(f"baseline:  DP gap={rep['dp_gap_full']:.4f}  "
          f"EO gap={rep['eo_gap_full']:.4f}  acc={rep['accuracy_full']:.4f}")
    print(f"admitted (n={rep['n_admitted']}/{rep['n_total']}):  "
          f"DP gap={rep['dp_gap_admitted']:.4f}  "
          f"EO gap={rep['eo_gap_admitted']:.4f}  acc={rep['accuracy_admitted']:.4f}")
    print(f"accuracy cost: {rep['accuracy_cost_pp']:.2f} pp")

    OUT.mkdir(parents=True, exist_ok=True)
    md = f"""# CST experiment results

Config: n={N}, seed={SEED}, tau_dp={TAU_DP}, tau_eo={TAU_EO}, eps={EPS}, delta={DELTA}, threshold={threshold:.4f}.

| metric | baseline (full set) | admitted subset |
|---|---|---|
| demographic-parity gap | {rep['dp_gap_full']:.4f} | {rep['dp_gap_admitted']:.4f} |
| equal-opportunity gap | {rep['eo_gap_full']:.4f} | {rep['eo_gap_admitted']:.4f} |
| accuracy | {rep['accuracy_full']:.4f} | {rep['accuracy_admitted']:.4f} |
| n decisions | {rep['n_total']} | {rep['n_admitted']} ({100*rep['admission_rate']:.1f}%) |

Accuracy cost of the fairness construction: **{rep['accuracy_cost_pp']:.2f} percentage points**.
"""
    (OUT / "results.md").write_text(md)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(2)
    axes[0].bar(x - 0.2, [rep["dp_gap_full"], rep["eo_gap_full"]], 0.4,
                label="baseline")
    axes[0].bar(x + 0.2, [rep["dp_gap_admitted"], rep["eo_gap_admitted"]], 0.4,
                label="admitted subset")
    axes[0].axhline(TAU_DP, color="red", linestyle="--", label="tau bound")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["demographic parity gap", "equal opportunity gap"])
    axes[0].set_ylabel("gap")
    axes[0].set_title("Fairness gaps: baseline vs admitted subset")
    axes[0].legend()

    axes[1].bar(["baseline", "admitted subset"],
                [rep["accuracy_full"], rep["accuracy_admitted"]])
    axes[1].set_ylabel("accuracy")
    axes[1].set_title(f"Accuracy (cost: {rep['accuracy_cost_pp']:.2f} pp)")
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUT / "charts.png", dpi=120)
    plt.close(fig)
    print(f"wrote {OUT / 'results.md'} and {OUT / 'charts.png'}")
    return rep


if __name__ == "__main__":
    main()
