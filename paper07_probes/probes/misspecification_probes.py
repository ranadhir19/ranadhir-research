"""Misspecification / false-ratification controls for Paper 7 (builder).

The released benchmark only ever shows the framework data generated
from its own model. These probes ask what the decision rule does when
the world disagrees with the model:
  A. All-null battery: all 19 designs with beta* = 0 -- how often does
     the governance rule falsely RATIFY noise? (seeded, 5 replicates)
  B. Heavy-tailed noise: benchmark designs, errors ~ t_3 (rescaled).
  C. Nonlinear truth: y includes a quadratic term the model omits.
Decisions use the released rule: accept iff BF10 > 3 and P_d > 0.95;
reject iff BF10 < 1/3; else inconclusive.
"""
import json
import numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location(
    "orig", "/home/hatch/workspace/goals/10-journal-papers-submitted-to-good-journals/hidden_files/paper07_work/probes/original_benchmark_copy.py")
orig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orig)
BENCH = orig.BENCH

def run_custom(beta_star, n, xlo, xhi, sigma, d, seed, noise="gauss", quad=0.0):
    rng = np.random.RandomState(seed)
    x = rng.uniform(xlo, xhi, n)
    xc0 = x - x.mean()
    if noise == "gauss":
        e = rng.normal(0, sigma, n)
    elif noise == "t3":
        e = rng.standard_t(3, n) * sigma / np.sqrt(3.0)  # unit-variance t3
    y = 2.0 + beta_star * xc0 + quad * (xc0 ** 2) / max(np.std(x), 1e-9) + e
    # reuse the released sampler pieces by monkey-running its log_post path
    import numpy as _np
    # replicate run_edge but on custom (x, y)
    def log_post(theta, x, y, sigma_beta):
        a, b, s = theta
        if s <= 0: return -1e300
        yhat = a + b * (x - x.mean())
        nn = len(x)
        ll = -0.5 * nn * _np.log(2 * _np.pi * s * s) - 0.5 * _np.sum((y - yhat) ** 2) / s / s
        lp = (-0.5 * (a / orig.PRIOR_ALPHA_S) ** 2 - 0.5 * (b / sigma_beta) ** 2
              - 0.5 * (s / orig.PRIOR_SIGMA_S) ** 2)
        return ll + lp
    xc = x - x.mean()
    b_ols = np.sum(xc * y) / np.sum(xc * xc)
    a_ols = y.mean(); s_ols = max(np.std(y - a_ols - b_ols * xc, ddof=2), 1e-3)
    theta = np.array([a_ols, b_ols, s_ols])
    scales = np.array([abs(a_ols) * 0.1 + 0.1, max(abs(b_ols) * 0.05, 0.05), max(s_ols * 0.05, 0.05)])
    B, D = orig.B if hasattr(orig, "B") else 2000, 4000
    B, D = 2000, 4000
    samples = np.empty((D, 3)); cur = log_post(theta, x, y, 5.0)
    rec = 0; recn = 0
    for k in range(B + D):
        prop = theta + rng.normal(0, scales); plp = log_post(prop, x, y, 5.0)
        if np.log(rng.rand()) < plp - cur: theta, cur = prop, plp; rec += 1
        recn += 1
        if k < B and (k + 1) % 200 == 0:
            ar = rec / recn
            if ar < 0.234: scales *= 0.9
            elif ar > 0.50: scales *= 1.1
            rec = 0; recn = 0
        if k >= B: samples[k - B] = theta
    beta = samples[:, 1]; bm, bs = float(beta.mean()), float(beta.std())
    lo, hi = orig.hdi94(beta)
    pdir = float(np.mean(np.sign(beta) == d))
    pri0 = (1 / (np.sqrt(2 * np.pi) * 5.0))
    pos0 = (1 / (np.sqrt(2 * np.pi) * max(bs, 1e-12))) * np.exp(-0.5 * (bm / max(bs, 1e-12)) ** 2)
    bf10 = float(pri0 / max(pos0, 1e-300))
    dec = "accept" if (bf10 > 3 and pdir > 0.95) else ("reject" if bf10 < 1/3 else "inconclusive")
    return dict(decision=dec, post_mean=bm, hdi_lo=lo, hdi_hi=hi, bf10=min(bf10, 1e6))

out = {}
# A. all-null battery, 5 replicates x 19 designs
null_counts = {"accept": 0, "reject": 0, "inconclusive": 0}
for rep in range(5):
    for i, (slug, bs, n, xlo, xhi, sig, d) in enumerate(BENCH):
        r = run_custom(0.0, n, xlo, xhi, sig, d, seed=5000 + rep * 100 + i)
        null_counts[r["decision"]] += 1
out["all_null_battery"] = dict(n_runs=95, **null_counts)
print("all-null battery (95 runs):", null_counts)
# B. heavy tails on 6 representative designs
t3 = {}
for i in (0, 2, 9, 13, 16, 18):
    slug, bs, n, xlo, xhi, sig, d = BENCH[i]
    r = run_custom(bs, n, xlo, xhi, sig, d, seed=6000 + i, noise="t3")
    t3[slug] = dict(beta_star=bs, **r, hdi_covers=bool(r["hdi_lo"] <= bs <= r["hdi_hi"]))
    print("t3", slug, r["decision"], "covers:", t3[slug]["hdi_covers"])
out["heavy_tail_t3"] = t3
# C. quadratic truth (standardised quad coefficient 1.0 * sigma scale)
qq = {}
for i in (0, 2, 9, 13, 18):
    slug, bs, n, xlo, xhi, sig, d = BENCH[i]
    r = run_custom(bs, n, xlo, xhi, sig, d, seed=7000 + i, quad=1.0 * sig)
    qq[slug] = dict(beta_star=bs, **r)
    print("quad", slug, r["decision"], "post_mean:", round(r["post_mean"], 3), "beta*:", bs)
out["quadratic_truth"] = qq
with open("misspecification_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("wrote misspecification_results.json")
