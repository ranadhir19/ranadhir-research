#!/usr/bin/env python3
"""Paper 7 cycle 2 severity suite, per upgrade2_preregistration.md
Section 4. Tester R3 variant spec (V1-V6), 30 worlds each, builder
code. Output: severity2_results.json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
sys.path.insert(0, str(WORK / "upgrade_probes"))
sys.path.insert(0, str(WORK / "probes"))
sys.path.insert(0, str(HERE))
from adjusted_core import fit_adjusted  # noqa: E402
from corrected_core import fit_corrected, decide_corrected, lambda_min  # noqa: E402
from rule_of_record import decide  # noqa: E402

LAM_GRID = (1.0, 0.9, 0.8, 0.7, 0.6)
LAM_TRUE_V3 = 0.64


def run_variant(seed, n_worlds=30, n_obs=180, conf_load=(0.50, 0.80),
                beta_range=(0.25, 0.60), proxy_noise=0.0, unmeasured=False,
                corr_true_cov=0.0, label=""):
    rng = np.random.default_rng(seed)
    worlds = []
    lam_min_counts = {}
    for wi in range(n_worlds):
        kinds = ["true"] * 16 + ["null"] * 12 + ["conf"] * 12
        order = rng.permutation(kinds)
        edges = []
        for kind in order:
            kind = str(kind)
            if kind == "true":
                c = rng.standard_normal(n_obs)
                if corr_true_cov > 0:
                    x = corr_true_cov * c + np.sqrt(1 - corr_true_cov ** 2) * rng.standard_normal(n_obs)
                else:
                    x = rng.standard_normal(n_obs)
                beta = rng.uniform(*beta_range); gamma = rng.uniform(-0.40, 0.40)
                y = beta * x + gamma * c + rng.standard_normal(n_obs); cov = c
            elif kind == "null":
                c = rng.standard_normal(n_obs); x = rng.standard_normal(n_obs)
                gamma = rng.uniform(-0.40, 0.40); y = gamma * c + rng.standard_normal(n_obs); cov = c
            else:
                z = rng.standard_normal(n_obs)
                lx, ly = rng.uniform(*conf_load), rng.uniform(*conf_load)
                x = lx * z + np.sqrt(1 - lx ** 2) * rng.standard_normal(n_obs)
                y = ly * z + np.sqrt(1 - ly ** 2) * rng.standard_normal(n_obs)
                if unmeasured:
                    cov = rng.standard_normal(n_obs)
                elif proxy_noise > 0:
                    cov = z + proxy_noise * rng.standard_normal(n_obs)
                else:
                    cov = z
            s1 = fit_adjusted(y, x, [], 1)
            s2 = fit_adjusted(y, x, [cov], 1)
            d1 = decide(s1["BF10"], s1["Pd"])
            d2 = decide(s2["BF10"], s2["Pd"])
            e = {"kind": kind, "rat_a": d1 == "accept",
                 "rat_b": d1 == "accept" and d2 == "accept"}
            if proxy_noise > 0:  # V3 only: procedures (d) and (e)
                sd_or = fit_corrected(y, x, [cov], 1, LAM_TRUE_V3)
                e["rat_d"] = d1 == "accept" and decide_corrected(sd_or) == "accept"
                dec_by_lam = {1.0: d2}
                for lam in LAM_GRID[1:]:
                    dec_by_lam[lam] = decide_corrected(fit_corrected(y, x, [cov], 1, lam))
                lm = lambda_min(dec_by_lam)
                e["lambda_min"] = lm
                e["rat_e"] = d1 == "accept" and lm == 0.6
                key = "none" if lm is None else str(lm)
                lam_min_counts[key] = lam_min_counts.get(key, 0) + 1
            edges.append(e)
        rec = {"world": wi}
        procs = ["a", "b"] + (["d", "e"] if proxy_noise > 0 else [])
        for p in procs:
            rat = [ee for ee in edges if ee[f"rat_{p}"]]
            false = [ee for ee in rat if ee["kind"] != "true"]
            rec[p] = {"ratified": len(rat), "false": len(false),
                      "true_ratified": sum(ee["kind"] == "true" for ee in rat),
                      "fdp": len(false) / len(rat) if rat else 0.0}
        worlds.append(rec)
        print(label, "world", wi, {p: (rec[p]["ratified"], rec[p]["false"]) for p in procs}, flush=True)

    def agg(proc, f):
        v = np.array([r[proc][f] for r in worlds], float)
        return {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}
    procs = ["a", "b"] + (["d", "e"] if proxy_noise > 0 else [])
    return {"label": label, "n_worlds": n_worlds,
            "summary": {p: {f: agg(p, f) for f in ("ratified", "false", "true_ratified", "fdp")} for p in procs},
            "lambda_min_counts_all_edges": lam_min_counts if proxy_noise > 0 else None,
            "worlds": worlds}


def main():
    out = {}
    out["V1_weak_confounding"] = run_variant(20261101, conf_load=(0.25, 0.45), label="V1")
    out["V2_n60"] = run_variant(20261102, n_obs=60, label="V2")
    out["V3_noisy_proxy_sd0.75"] = run_variant(20261103, proxy_noise=0.75, label="V3")
    out["V4_unmeasured"] = run_variant(20261104, unmeasured=True, label="V4")
    out["V5_weak_true_beta"] = run_variant(20261105, beta_range=(0.10, 0.30), label="V5")
    out["V6_corr_true_cov_rho0.7"] = run_variant(20261106, corr_true_cov=0.7, label="V6")
    (HERE / "severity2_results.json").write_text(json.dumps(out, indent=2))
    for k, v in out.items():
        print(k, {p: (round(v["summary"][p]["ratified"]["mean"], 2),
                      round(v["summary"][p]["false"]["mean"], 2),
                      round(v["summary"][p]["true_ratified"]["mean"], 2),
                      round(v["summary"][p]["fdp"]["mean"], 3)) for p in v["summary"]})


if __name__ == "__main__":
    main()
