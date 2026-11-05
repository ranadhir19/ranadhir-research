#!/usr/bin/env python3
"""Paper 7 cycle 2 large-ledger experiment, per
upgrade2_preregistration.md Section 5. 40 worlds x 120 edges.
Output: large_ledger_results.json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
sys.path.insert(0, str(WORK / "upgrade_probes"))
sys.path.insert(0, str(WORK / "probes"))
from adjusted_core import fit_adjusted, p_claim, ledger_prefix  # noqa: E402
from rule_of_record import decide  # noqa: E402

SEED = 20261201
N_WORLDS, M = 40, 120
N_TRUE, N_NULL, N_CONF = 48, 36, 36


def gen_edge(rng, kind, n):
    if kind == "true":
        c = rng.standard_normal(n)
        x = rng.standard_normal(n)
        beta = rng.uniform(0.15, 0.60)
        gamma = rng.uniform(-0.40, 0.40)
        y = beta * x + gamma * c + rng.standard_normal(n)
        return x, y, c
    if kind == "null":
        c = rng.standard_normal(n)
        x = rng.standard_normal(n)
        gamma = rng.uniform(-0.40, 0.40)
        y = gamma * c + rng.standard_normal(n)
        return x, y, c
    z = rng.standard_normal(n)
    lx, ly = rng.uniform(0.50, 0.80), rng.uniform(0.50, 0.80)
    x = lx * z + np.sqrt(1 - lx ** 2) * rng.standard_normal(n)
    y = ly * z + np.sqrt(1 - ly ** 2) * rng.standard_normal(n)
    return x, y, z


def main():
    rng = np.random.default_rng(SEED)
    kinds_base = ["true"] * N_TRUE + ["null"] * N_NULL + ["conf"] * N_CONF
    worlds = []
    totals = {"c05_removed_false": 0, "c05_removed_true": 0,
              "c01_removed_false": 0, "c01_removed_true": 0,
              "worlds_c05_differs": 0, "worlds_c01_differs": 0}
    for wi in range(N_WORLDS):
        order = rng.permutation(kinds_base)
        edges = []
        for kind in order:
            kind = str(kind)
            n = int(rng.choice([60, 120, 180]))
            x, y, c = gen_edge(rng, kind, n)
            s1 = fit_adjusted(y, x, [], +1)
            s2 = fit_adjusted(y, x, [c], +1)
            d1 = decide(s1["BF10"], s1["Pd"])
            d2 = decide(s2["BF10"], s2["Pd"])
            pc05, _ = p_claim(s2["BF10"], s2["Pd"], 0.5, log_bf10=s2["log_BF10"])
            pc01, _ = p_claim(s2["BF10"], s2["Pd"], 0.1, log_bf10=s2["log_BF10"])
            edges.append({"kind": kind, "n": n, "d1": d1, "d2": d2,
                          "rat_a": d1 == "accept",
                          "rat_b": d1 == "accept" and d2 == "accept",
                          "p_claim_05": pc05, "p_claim_01": pc01})
        rat_b_idx = [i for i, e in enumerate(edges) if e["rat_b"]]
        for wkey, proc, pkey in (("0.5", "c05", "p_claim_05"), ("0.1", "c01", "p_claim_01")):
            keep = ledger_prefix([{"p_claim": edges[i][pkey]} for i in rat_b_idx])
            kept_global = {rat_b_idx[j] for j in keep}
            for i in rat_b_idx:
                edges[i][f"rat_{proc}"] = i in kept_global
            removed = [edges[i] for i in rat_b_idx if i not in kept_global]
            totals[f"{proc}_removed_false"] += sum(e["kind"] != "true" for e in removed)
            totals[f"{proc}_removed_true"] += sum(e["kind"] == "true" for e in removed)
            if removed:
                totals[f"worlds_{proc}_differs"] += 1
        for e in edges:
            e.setdefault("rat_c05", False); e.setdefault("rat_c01", False)
        rec = {"world": wi}
        for proc in ("a", "b", "c05", "c01"):
            rat = [e for e in edges if e[f"rat_{proc}"]]
            false = [e for e in rat if e["kind"] != "true"]
            rec[proc] = {"ratified": len(rat), "false": len(false),
                         "true_ratified": sum(e["kind"] == "true" for e in rat),
                         "fdp": len(false) / len(rat) if rat else 0.0}
        b_rat = [e for e in edges if e["rat_b"]]
        rec["post_exp_fdp_b_w05"] = float(np.mean([1 - e["p_claim_05"] for e in b_rat])) if b_rat else 0.0
        rec["realised_fdp_b"] = rec["b"]["fdp"]
        worlds.append(rec)
        print("world", wi, {p: (rec[p]["ratified"], rec[p]["false"], rec[p]["true_ratified"]) for p in ("a", "b", "c05", "c01")},
              "postExpFDP_b", round(rec["post_exp_fdp_b_w05"], 4), flush=True)

    def summ(proc, f):
        v = np.array([w[proc][f] for w in worlds], float)
        return {"mean": float(v.mean()), "median": float(np.median(v)),
                "min": float(v.min()), "max": float(v.max())}
    pe = np.array([w["post_exp_fdp_b_w05"] for w in worlds])
    rf = np.array([w["realised_fdp_b"] for w in worlds])
    out = {"preregistration": "upgrade2_preregistration.md Section 5",
           "spec": {"worlds": N_WORLDS, "edges_per_world": M, "master_seed": SEED,
                    "composition": {"true": N_TRUE, "null": N_NULL, "confounded": N_CONF},
                    "n_per_edge_drawn_from": [60, 120, 180], "beta_true": [0.15, 0.60],
                    "ledger_gamma": 0.05},
           "summary": {p: {f: summ(p, f) for f in ("ratified", "false", "true_ratified", "fdp")}
                       for p in ("a", "b", "c05", "c01")},
           "decisive": totals,
           "calibration_b_set": {"posterior_expected_fdp_mean": float(pe.mean()),
                                 "realised_fdp_mean": float(rf.mean()),
                                 "mean_difference_post_minus_realised": float((pe - rf).mean())},
           "worlds": worlds}
    (HERE / "large_ledger_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["summary"], indent=1))
    print("decisive:", totals)
    print("calibration:", out["calibration_b_set"])


if __name__ == "__main__":
    main()
