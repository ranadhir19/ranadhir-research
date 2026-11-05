"""Reproduce the 19-edge Bayesian KPI Governance benchmark (Paper #7 revision).

Each edge is a known ground-truth slope beta_star; we generate (x,y) under
y = alpha + beta_star*(x-xbar) + N(0, sigma^2), then run a self-contained
random-walk Metropolis-Hastings sampler and compute Savage-Dickey BF10.

Reports per edge: posterior mean beta, 94% HDI, P(sign=expected),
BF10, decision (accept/reject/inconclusive), MH acceptance rate.

Honest extension over the original paper:
  - reports relative-error per edge so the "mean rel-err <=9% for 15/18"
    claim becomes machine-checkable;
  - reports BF saturation count;
  - reports HDI-coverage of beta_star.
"""
import os, json, time
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "kpi_governance_results.json")

# 19-edge benchmark (slug, beta_star, n, x_lo, x_hi, sigma, expected_dir)
BENCH = [
    ("itsm_p1_drives_churn",          0.80, 180, 0,  10, 1.0, +1),
    ("dora_freq_lifts_csat",          0.60, 200, 0,  20, 1.5, +1),
    ("dso_drains_runway",            -0.50, 150, 30, 90, 2.0, -1),
    ("eng_investment_lifts_velocity", 0.45, 180, 0,  50, 2.5, +1),
    ("rework_hurts_velocity",        -0.70, 160, 0,  30, 2.0, -1),
    ("training_hours_lift_engagement",0.30, 140, 0,  60, 2.5, +1),
    ("attrition_hurts_velocity",     -0.90, 170, 0,  20, 1.5, -1),
    ("mttr_hurts_csat",              -0.55, 190, 1,  48, 1.8, -1),
    ("backlog_drives_lateness",       0.65, 150, 0,  40, 2.0, +1),
    ("uptime_lifts_csat",             0.40, 200, 95, 100,1.0, +1),
    ("supplier_otd_drives_otif",      0.85, 160, 70, 100,1.5, +1),
    ("scope_creep_hurts_margin",     -0.50, 140, 0,  30, 2.0, -1),
    ("ad_spend_drives_pipeline",      0.55, 180, 0,  50, 3.0, +1),
    ("nps_drives_renewal",            0.75, 170, -50,80, 2.0, +1),
    ("tech_debt_hurts_velocity",     -0.60, 160, 0,  40, 2.5, -1),
    ("hiring_lag_hurts_capacity",    -0.40, 150, 0,  90, 2.5, -1),
    ("pmo_bottleneck_hits_velocity", -22.0, 160, 0,  0.12, 6.0, -1),  # the tricky one
    ("incidents_per_release",         0.20, 200, 0, 100, 1.5, +1),
    ("null_demo",                     0.00, 180, 0,  10, 1.0, +1),
]

PRIOR_ALPHA_S = 10.0
PRIOR_SIGMA_S = 5.0

def log_post(theta, x, y, sigma_beta):
    a, b, s = theta
    if s <= 0: return -1e300
    yhat = a + b*(x - x.mean())
    n = len(x)
    ll = -0.5*n*np.log(2*np.pi*s*s) - 0.5*np.sum((y - yhat)**2)/s/s
    lp = (-0.5*(a/PRIOR_ALPHA_S)**2
          -0.5*(b/sigma_beta)**2
          -0.5*(s/PRIOR_SIGMA_S)**2)  # half-normal (s>0 enforced)
    return ll + lp

def hdi94(samples):
    s = np.sort(samples)
    n = len(s); k = int(0.94*n)
    width = s[k:] - s[:n-k]
    i = int(np.argmin(width))
    return float(s[i]), float(s[i+k])

def run_edge(beta_star, n, xlo, xhi, sigma, expected_dir, sigma_beta=5.0,
             B=2000, D=4000, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.uniform(xlo, xhi, n)
    y = 2.0 + beta_star*(x - x.mean()) + rng.normal(0, sigma, n)
    # OLS warm start
    xc = x - x.mean()
    b_ols = np.sum(xc*y)/np.sum(xc*xc) if np.sum(xc*xc) > 1e-12 else 0.0
    a_ols = y.mean()
    s_ols = max(np.std(y - a_ols - b_ols*xc, ddof=2), 1e-3)
    theta = np.array([a_ols, b_ols, s_ols])
    scales = np.array([abs(a_ols)*0.1+0.1,
                       max(abs(b_ols)*0.05, 0.05),
                       max(s_ols*0.05, 0.05)])
    samples = np.empty((D, 3))
    accepts = 0; recent = 0; recent_n = 0
    cur_lp = log_post(theta, x, y, sigma_beta)
    for k in range(B+D):
        prop = theta + rng.normal(0, scales)
        prop_lp = log_post(prop, x, y, sigma_beta)
        if np.log(rng.rand()) < prop_lp - cur_lp:
            theta = prop; cur_lp = prop_lp; accepts += 1; recent += 1
        recent_n += 1
        if k < B and (k+1) % 200 == 0:
            ar = recent / recent_n
            if ar < 0.234: scales *= 0.9
            elif ar > 0.50: scales *= 1.1
            recent = 0; recent_n = 0
        if k >= B: samples[k-B] = theta
    beta = samples[:, 1]
    bm, bs = float(beta.mean()), float(beta.std())
    lo, hi = hdi94(beta)
    pdir = float(np.mean(np.sign(beta) == expected_dir))
    # Savage-Dickey: BF10 = prior(0)/posterior(0) under Gaussian approx
    pri0 = (1/(np.sqrt(2*np.pi)*sigma_beta))*np.exp(-0.5*(0/sigma_beta)**2)
    pos0 = (1/(np.sqrt(2*np.pi)*max(bs,1e-12)))*np.exp(-0.5*(bm/max(bs,1e-12))**2)
    # Savage-Dickey: BF10 = prior(beta=0) / posterior(beta=0)
    bf10 = float(pri0/max(pos0, 1e-300))
    bf_disp = min(bf10, 1e6)
    if beta_star == 0.0:
        decision = "reject" if bf10 < 1/3 else ("inconclusive" if bf10 < 3 else "accept")
    else:
        decision = "accept" if (bf10 > 3 and pdir > 0.95) else (
                  "reject" if bf10 < 1/3 else "inconclusive")
    return dict(post_mean=bm, post_sd=bs, hdi_lo=lo, hdi_hi=hi,
                p_dir=pdir, bf10=bf_disp, bf10_saturated=(bf10 >= 1e6),
                decision=decision, accept_rate=accepts/(B+D))

def main():
    t0 = time.time()
    edges = []
    for i, (slug, bs, n, xlo, xhi, sig, d) in enumerate(BENCH):
        r = run_edge(bs, n, xlo, xhi, sig, d, seed=i)
        rel_err = (None if bs == 0 else abs((r["post_mean"]-bs)/bs))
        in_hdi  = (r["hdi_lo"] <= bs <= r["hdi_hi"])
        r.update(slug=slug, beta_star=bs, rel_err=rel_err, hdi_covers=in_hdi)
        edges.append(r)
        print(f"{slug:34s} bs={bs:7.3f} bm={r['post_mean']:7.3f} "
              f"HDI=[{r['hdi_lo']:7.3f},{r['hdi_hi']:7.3f}] "
              f"BF10={r['bf10']:.2e} dec={r['decision']:8s} ar={r['accept_rate']:.2f}")
    nonnull = [e for e in edges if e["beta_star"] != 0]
    rels = [e["rel_err"] for e in nonnull]
    summary = dict(
        generated=time.strftime("%Y-%m-%d %H:%M"),
        n_edges=len(edges),
        n_nonnull=len(nonnull),
        accept_rate_min=min(e["accept_rate"] for e in edges),
        accept_rate_max=max(e["accept_rate"] for e in edges),
        accept_rate_mean=float(np.mean([e["accept_rate"] for e in edges])),
        n_within_optimal_band=sum(0.234 <= e["accept_rate"] <= 0.50 for e in edges),
        rel_err_mean=float(np.mean(rels)),
        rel_err_median=float(np.median(rels)),
        n_rel_err_le_9pct=sum(r <= 0.09 for r in rels),
        n_hdi_covers=sum(e["hdi_covers"] for e in nonnull),
        n_bf_saturated=sum(e["bf10_saturated"] for e in edges),
        null_edge_decision=[e["decision"] for e in edges if e["beta_star"]==0][0],
        null_edge_bf01=1.0/[e["bf10"] for e in edges if e["beta_star"]==0][0],
        wall_time_sec=round(time.time()-t0, 2),
    )
    out = dict(summary=summary, edges=edges)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT}")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
