"""TESTER Probe B (independent controls the builder did NOT run) for Paper 7.

Sampler: my own re-implementation of the released protocol (OLS warm
start, adaptive RW-MH, B=2000, D=4000, sigma_beta=5, Gaussian plug-in
Savage-Dickey), written from the manuscript's Sec. Method, applied to
custom (x, y) the released run_edge cannot generate.

B1 CONFOUNDER: z drives both x and y; true direct effect beta=0.
   Does the governance rule ratify a causally false edge? (5 seeds)
B2 WRONG DIRECTION: true beta=-0.8, elicited direction +1. (3 seeds)
B3 HETEROSCEDASTIC noise: sigma_i grows with |x - xbar|. itsm + null.
B4 PRIOR SENSITIVITY on a WEAK NON-NULL edge (uptime design): the
   manuscript shows prior sensitivity only for the null edge.
B5 DECISION-RULE FIDELITY: published rule (manuscript Table 1) vs the
   rule implemented in the released code, enumerated on a grid; plus a
   check that no realised edge in any shipped JSON falls in the
   divergence region.
"""
import json
import numpy as np

OUT = "/home/hatch/workspace/goals/10-journal-papers-submitted-to-good-journals/hidden_files/paper07_work/tester_probes/probeB_results.json"
PA_S, PS_S = 10.0, 5.0

def run_xy(x, y, expected_dir, sigma_beta=5.0, seed=0, B=2000, D=4000):
    rng = np.random.RandomState(seed)
    xc = x - x.mean(); n = len(x)
    b_ols = np.sum(xc * y) / np.sum(xc * xc); a_ols = y.mean()
    s_ols = max(np.std(y - a_ols - b_ols * xc, ddof=2), 1e-3)
    th = np.array([a_ols, b_ols, s_ols])
    sc = np.array([abs(a_ols) * 0.1 + 0.1, max(abs(b_ols) * 0.05, 0.05), max(s_ols * 0.05, 0.05)])
    def lp(t):
        a, b, s = t
        if s <= 0: return -1e300
        r = y - a - b * xc
        return (-0.5 * n * np.log(2 * np.pi * s * s) - 0.5 * np.sum(r * r) / s / s
                - 0.5 * (a / PA_S) ** 2 - 0.5 * (b / sigma_beta) ** 2 - 0.5 * (s / PS_S) ** 2)
    cur = lp(th); samp = np.empty((D, 3)); rec = recn = 0
    for k in range(B + D):
        prop = th + rng.normal(0, sc); pl = lp(prop)
        if np.log(rng.rand()) < pl - cur: th, cur = prop, pl; rec += 1
        recn += 1
        if k < B and (k + 1) % 200 == 0:
            ar = rec / recn
            if ar < 0.234: sc *= 0.9
            elif ar > 0.50: sc *= 1.1
            rec = recn = 0
        if k >= B: samp[k - B] = th
    beta = samp[:, 1]; bm, bs = float(beta.mean()), float(beta.std())
    srt = np.sort(beta); kk = int(0.94 * D); w = srt[kk:] - srt[:D - kk]; i = int(np.argmin(w))
    lo, hi = float(srt[i]), float(srt[i + kk])
    pdir = float(np.mean(np.sign(beta) == expected_dir))
    pri0 = 1 / (np.sqrt(2 * np.pi) * sigma_beta)
    pos0 = 1 / (np.sqrt(2 * np.pi) * max(bs, 1e-12)) * np.exp(-0.5 * (bm / max(bs, 1e-12)) ** 2)
    bf10 = float(pri0 / max(pos0, 1e-300))
    dec_code = "accept" if (bf10 > 3 and pdir > 0.95) else ("reject" if bf10 < 1 / 3 else "inconclusive")
    dec_paper = ("accept" if (bf10 >= 10 and pdir >= 0.95) or (3 <= bf10 < 10 and pdir >= 0.90)
                 else ("reject" if bf10 <= 1 / 3 else "inconclusive"))
    return dict(post_mean=bm, post_sd=bs, hdi_lo=lo, hdi_hi=hi, p_dir=pdir,
                bf10=min(bf10, 1e6), decision_code_rule=dec_code, decision_paper_rule=dec_paper)

res = {}
# B1 confounder: x = 0.8 z + e_x ; y = 1.5 z + e_y  (direct effect of x on y is 0)
b1 = []
for seed in range(5):
    rng = np.random.RandomState(9000 + seed)
    z = rng.uniform(0, 10, 180)
    x = 0.8 * z + rng.normal(0, 0.7, 180)
    y = 2.0 + 1.5 * z + rng.normal(0, 1.0, 180)
    r = run_xy(x, y, +1, seed=seed)
    b1.append(r); print("B1 confounder seed", seed, r["decision_code_rule"], "bhat=%.3f" % r["post_mean"], "BF10=%.3g" % r["bf10"])
res["B1_confounder_true_direct_effect_0"] = b1
# B2 wrong direction: itsm design, beta*=-0.8, elicited +1
b2 = []
for seed in range(3):
    rng = np.random.RandomState(9100 + seed)
    x = rng.uniform(0, 10, 180)
    y = 2.0 - 0.8 * (x - x.mean()) + rng.normal(0, 1.0, 180)
    r = run_xy(x, y, +1, seed=seed)
    b2.append(r); print("B2 wrong-dir seed", seed, r["decision_code_rule"], "/", r["decision_paper_rule"], "Pd=%.3f" % r["p_dir"])
res["B2_wrong_direction"] = b2
# B3 heteroscedastic: sigma_i = sigma * (0.4 + |xc|/sd(x)); itsm design + null design
b3 = {}
for name, bs_true, xlo, xhi, sg in (("itsm_like", 0.8, 0, 10, 1.0), ("null_like", 0.0, 0, 10, 1.0)):
    rng = np.random.RandomState(9200)
    x = rng.uniform(xlo, xhi, 180); xc = x - x.mean()
    sig_i = sg * (0.4 + np.abs(xc) / np.std(x))
    y = 2.0 + bs_true * xc + rng.normal(0, 1, 180) * sig_i
    r = run_xy(x, y, +1, seed=7)
    b3[name] = dict(beta_star=bs_true, **r, hdi_covers=bool(r["hdi_lo"] <= bs_true <= r["hdi_hi"]))
    print("B3", name, r["decision_code_rule"], "covers:", b3[name]["hdi_covers"])
res["B3_heteroscedastic"] = b3
# B4 prior sensitivity, weak non-null edge (uptime design: beta*=0.4, x in [95,100], sigma=1, n=200)
rng = np.random.RandomState(9)
x = rng.uniform(95, 100, 200)
y = 2.0 + 0.4 * (x - x.mean()) + rng.normal(0, 1.0, 200)
b4 = []
for sb in (1.0, 2.5, 5.0, 10.0, 25.0):
    r = run_xy(x, y, +1, sigma_beta=sb, seed=9)
    b4.append(dict(sigma_beta=sb, bf10=r["bf10"], decision=r["decision_code_rule"], post_mean=r["post_mean"]))
    print("B4 uptime-like sigma_beta", sb, "BF10=%.4g" % r["bf10"], r["decision_code_rule"])
res["B4_prior_sens_weak_nonnull_uptime_seed9"] = b4
# B5 rule fidelity grid
div = []
for bf in [0.1, 0.34, 1.0, 2.9, 3.0, 5.0, 9.9, 10.0, 100.0]:
    for pd in [0.50, 0.89, 0.90, 0.92, 0.949, 0.95, 0.99]:
        paper = ("accept" if (bf >= 10 and pd >= 0.95) or (3 <= bf < 10 and pd >= 0.90)
                 else ("reject" if bf <= 1 / 3 else "inconclusive"))
        code = "accept" if (bf > 3 and pd > 0.95) else ("reject" if bf < 1 / 3 else "inconclusive")
        if paper != code: div.append(dict(bf10=bf, p_dir=pd, paper=paper, code=code))
res["B5_divergent_grid_cells"] = div
print("B5 divergent cells:", len(div), div[:6])
# realised edges in divergence region?
ship = json.load(open("/home/hatch/workspace/github-fis-watch/mirrors-mcp/journal/07_Bayesian_KPI_Governance/experiments/kpi_governance_results.json"))
realised = [(e["slug"], e["bf10"], e["p_dir"]) for e in ship["edges"]
            if 3 <= e["bf10"] < 10 or (e["bf10"] >= 10 and 0.90 <= e["p_dir"] < 0.96)]
res["B5_realised_edges_near_divergence"] = realised
print("B5 realised edges in/near divergence region:", realised)
json.dump(res, open(OUT, "w"), indent=2)
print("wrote", OUT)
