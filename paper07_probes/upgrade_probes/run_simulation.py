#!/usr/bin/env python3
"""Paper 7 upgrade: ledger-level simulation calibration, per
upgrade_preregistration.md section 4. 120 worlds x 40 edges (16 true direct,
12 pure null, 12 confounded-only), n = 180, master seed 20261027.
Procedures: (a) stage-1 bivariate rule; (b) two-stage adjusted; (c) two-stage
+ ledger-FDR prefix rule (gamma = 0.05, w = 0.5).
Output: simulation_results.json -- the sole source of manuscript numbers.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "probes"))
from adjusted_core import fit_adjusted, p_claim, ledger_prefix  # noqa: E402
from rule_of_record import decide  # noqa: E402

N_WORLDS, M, N_OBS = 120, 40, 180
N_TRUE, N_NULL, N_CONF = 16, 12, 12
SEED = 20261027


def gen_edge(rng, kind):
    if kind == "true":
        c = rng.standard_normal(N_OBS)
        x = rng.standard_normal(N_OBS)
        beta = rng.uniform(0.25, 0.60)
        gamma = rng.uniform(-0.40, 0.40)
        y = beta * x + gamma * c + rng.standard_normal(N_OBS)
        return x, y, c
    if kind == "null":
        c = rng.standard_normal(N_OBS)
        x = rng.standard_normal(N_OBS)
        gamma = rng.uniform(-0.40, 0.40)
        y = gamma * c + rng.standard_normal(N_OBS)
        return x, y, c
    z = rng.standard_normal(N_OBS)
    lx, ly = rng.uniform(0.50, 0.80), rng.uniform(0.50, 0.80)
    x = lx * z + np.sqrt(1 - lx ** 2) * rng.standard_normal(N_OBS)
    y = ly * z + np.sqrt(1 - ly ** 2) * rng.standard_normal(N_OBS)
    return x, y, z


def main():
    rng = np.random.default_rng(SEED)
    kinds = (["true"] * N_TRUE + ["null"] * N_NULL + ["conf"] * N_CONF)
    worlds = []
    for wi in range(N_WORLDS):
        order = rng.permutation(kinds)
        edges = []
        for kind in order:
            x, y, c = gen_edge(rng, str(kind))
            s1 = fit_adjusted(y, x, [], +1)
            s2 = fit_adjusted(y, x, [c], +1)
            d1 = decide(s1["BF10"], s1["Pd"])
            d2 = decide(s2["BF10"], s2["Pd"])
            pc, _ = p_claim(s2["BF10"], s2["Pd"], 0.5, log_bf10=s2["log_BF10"])
            edges.append({"kind": str(kind), "d1": d1, "d2": d2,
                          "rat_a": d1 == "accept",
                          "rat_b": d1 == "accept" and d2 == "accept",
                          "p_claim": pc})
        keep = ledger_prefix([{"p_claim": e["p_claim"]} for e in edges if e["rat_b"]])
        rat_b_idx = [i for i, e in enumerate(edges) if e["rat_b"]]
        for j, i in enumerate(rat_b_idx):
            edges[i]["rat_c"] = j in keep
        for e in edges:
            e.setdefault("rat_c", False)
        rec = {"world": wi}
        for proc in ("a", "b", "c"):
            key = f"rat_{proc}"
            rat = [e for e in edges if e[key]]
            false_rat = [e for e in rat if e["kind"] != "true"]
            rec[proc] = {
                "ratified": len(rat),
                "false_ratified": len(false_rat),
                "confounded_ratified": sum(e["kind"] == "conf" for e in rat),
                "true_ratified": sum(e["kind"] == "true" for e in rat),
                "fdp": (len(false_rat) / len(rat)) if rat else 0.0,
            }
        worlds.append(rec)
        if (wi + 1) % 20 == 0:
            print(f"world {wi + 1}/{N_WORLDS}", flush=True)

    def summ(proc, field):
        v = np.array([w[proc][field] for w in worlds], float)
        return {"mean": float(v.mean()), "median": float(np.median(v)),
                "min": float(v.min()), "max": float(v.max())}

    out = {
        "preregistration": "upgrade_preregistration.md section 4 (frozen before this run)",
        "spec": {"worlds": N_WORLDS, "edges_per_world": M, "n_per_edge": N_OBS,
                 "composition": {"true": N_TRUE, "null": N_NULL, "confounded": N_CONF},
                 "master_seed": SEED, "ledger_gamma": 0.05, "ledger_w": 0.5},
        "summary": {p: {f: summ(p, f) for f in
                        ("ratified", "false_ratified", "confounded_ratified",
                         "true_ratified", "fdp")} for p in ("a", "b", "c")},
        "worlds": worlds,
    }
    (HERE / "simulation_results.json").write_text(json.dumps(out, indent=2))
    for p in ("a", "b", "c"):
        s = out["summary"][p]
        print(p, {f: round(s[f]["mean"], 3) for f in s})
    print("wrote", HERE / "simulation_results.json")


if __name__ == "__main__":
    main()
