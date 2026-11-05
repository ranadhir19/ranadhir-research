#!/usr/bin/env python3
"""Paper 7 strengthening items 2-3 verification runs.

(A) Multi-seed decisions under BOTH rules: re-runs the canonical sampler for
    10 master seeds x 19 edges (decisions only; bridge/R-hat already on record)
    and recomputes each seed's decision vector under the shipped code's rule
    and under the published rule of record (rule_of_record.decide).
(B) Confounder control (tester item 3): common cause z drives x and y, true
    direct effect exactly 0; 5 seeds through the shipped sampler + Savage-Dickey
    BF + rule of record. The governance-critical misspecification.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rule_of_record import decide  # noqa: E402

spec = importlib.util.spec_from_file_location("orig", HERE / "original_benchmark_copy.py")
orig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orig)

BENCH = orig.BENCH


def old_rule(bf, pd):
    return "accept" if (bf > 3 and pd > 0.95) else ("reject" if bf < 1 / 3 else "inconclusive")


def run_all_seeds():
    per_seed = []
    for master in range(10):
        vec_old, vec_new = {}, {}
        for i, (slug, bs, n, xlo, xhi, sig, d) in enumerate(BENCH):
            r = orig.run_edge(bs, n, xlo, xhi, sig, d, seed=master * 1000 + i)
            bf = r["bf10"] if not r["bf10_saturated"] else 1e6
            vec_old[slug] = old_rule(bf, r["p_dir"])
            vec_new[slug] = decide(bf, r["p_dir"])
        per_seed.append(dict(master_seed=master, old=vec_old, new=vec_new,
                             identical_rules_agree=vec_old == vec_new))
    base = per_seed[0]["new"]
    stable = all(s["new"] == base for s in per_seed)
    return per_seed, stable, base


def confounder_control():
    rows = []
    for s in range(5):
        rng = np.random.RandomState(777 + s)
        n = 180
        z = rng.normal(0, 1, n)
        x = z + rng.normal(0, 1, n)
        y = 2.0 + 2.0 * z + rng.normal(0, 2.0, n)  # direct effect of x on y: 0
        xc = x - x.mean()
        # shipped sampler design via field_study's run_chain on (alpha,beta,sigma)
        sys.path.insert(0, str(HERE))
        import field_study as fs
        draws, acc = fs.run_chain(xc, y, 5.0, 4242 + s, "m1")
        beta = draws[:, 1]
        bm, bsd = float(beta.mean()), float(beta.std())
        pri0 = 1 / (np.sqrt(2 * np.pi) * 5.0)
        pos0 = (1 / (np.sqrt(2 * np.pi) * max(bsd, 1e-12))) * np.exp(-0.5 * (bm / max(bsd, 1e-12)) ** 2)
        bf = float(pri0 / max(pos0, 1e-300))
        pd_ = float(np.mean(beta > 0))
        rows.append(dict(seed=s, beta_mean=bm, bf10=min(bf, 1e6), Pd=pd_,
                         decision=decide(bf, pd_), acceptance=acc))
    return rows


if __name__ == "__main__":
    t0 = time.time()
    per_seed, stable, base = run_all_seeds()
    print(f"[A] 10-seed decision vectors identical under rule of record: {stable}")
    print(f"[A] old-vs-new rule agreement per seed: {[s['identical_rules_agree'] for s in per_seed]}")
    print(f"[A] canonical vector: accept={list(base.values()).count('accept')} "
          f"inconclusive={list(base.values()).count('inconclusive')} reject={list(base.values()).count('reject')}")
    conf = confounder_control()
    for r in conf:
        print(f"[B] confounder seed {r['seed']}: beta={r['beta_mean']:+.3f} BF10={r['bf10']:.3g} "
              f"Pd={r['Pd']:.3f} -> {r['decision']} (true direct effect = 0)")
    out = dict(multiseed_rule_check=dict(stable_across_seeds=stable,
                                         rules_agree_per_seed=[s["identical_rules_agree"] for s in per_seed],
                                         canonical_vector=base),
               confounder_control=conf, wall_time_sec=time.time() - t0)
    (HERE / "strengthening_checks_results.json").write_text(json.dumps(out, indent=2))
    print("wrote strengthening_checks_results.json")
