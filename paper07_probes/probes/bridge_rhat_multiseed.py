"""Paper 7 strengthening experiments (builder).

Implements, on top of the released replication benchmark:
  (1) Bridge sampling (Meng & Wong 1996; Gronau et al. 2017) for the
      marginal likelihoods of M1 (beta free) and M0 (beta = 0), giving
      finite log BF10 for the edges where the Savage-Dickey Gaussian
      approximation saturates at the 1e6 display cap.
  (2) A deterministic quadrature ground truth for log p(y|M) (alpha
      integrated analytically; trapezoid grid over (beta, log sigma)
      for M1 and over log sigma for M0) used ONLY to validate the
      bridge sampler.
  (3) Four-chain rank-normalised split-Rhat (Vehtari et al. 2021) for
      beta on every edge.
  (4) Multi-seed robustness: the full benchmark under 10 master seeds.
  (5) Prior-scale sensitivity of the null edge's BF01 (Savage-Dickey
      and bridge) for sigma_beta in {1, 2.5, 5, 10, 25}.

All numbers are written to bridge_rhat_multiseed_results.json.
"""
import json, time
import numpy as np
from scipy import stats, special

# ---------------------------------------------------------------- benchmark
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
    ("pmo_bottleneck_hits_velocity", -22.0, 160, 0,  0.12, 6.0, -1),
    ("incidents_per_release",         0.20, 200, 0, 100, 1.5, +1),
    ("null_demo",                     0.00, 180, 0,  10, 1.0, +1),
]
PA, PS = 10.0, 5.0          # alpha prior sd, sigma half-normal scale
B, D = 2000, 4000

def gen_data(beta_star, n, xlo, xhi, sigma, seed):
    rng = np.random.RandomState(seed)
    x = rng.uniform(xlo, xhi, n)
    y = 2.0 + beta_star * (x - x.mean()) + rng.normal(0, sigma, n)
    return x, y

def loglik_ab(a, b, s, xc, y):
    r = y - a - b * xc
    n = len(y)
    return -0.5 * n * np.log(2 * np.pi * s * s) - 0.5 * np.sum(r * r) / (s * s)

def log_post_m1(theta, xc, y, sb):
    a, b, s = theta
    if s <= 0: return -1e300
    return (loglik_ab(a, b, s, xc, y)
            - 0.5 * (a / PA) ** 2 - 0.5 * (b / sb) ** 2 - 0.5 * (s / PS) ** 2)

def log_post_m0(theta, xc, y, sb):
    a, s = theta
    if s <= 0: return -1e300
    return (loglik_ab(a, 0.0, s, xc, y)
            - 0.5 * (a / PA) ** 2 - 0.5 * (s / PS) ** 2)

def ols(xc, y):
    sxx = np.sum(xc * xc)
    b = np.sum(xc * y) / sxx if sxx > 1e-12 else 0.0
    a = y.mean()
    s = max(np.std(y - a - b * xc, ddof=2), 1e-3)
    return a, b, s

def run_chain(xc, y, sb, seed, model="m1", rng=None):
    """Random-walk MH identical in design to the released benchmark."""
    if rng is None:
        rng = np.random.RandomState(seed)
    a0, b0, s0 = ols(xc, y)
    if model == "m1":
        theta = np.array([a0, b0, s0])
        scales = np.array([abs(a0) * 0.1 + 0.1, max(abs(b0) * 0.05, 0.05),
                           max(s0 * 0.05, 0.05)])
        lp = lambda th: log_post_m1(th, xc, y, sb)
    else:
        theta = np.array([a0, s0])
        scales = np.array([abs(a0) * 0.1 + 0.1, max(s0 * 0.05, 0.05)])
        lp = lambda th: log_post_m0(th, xc, y, sb)
    out = np.empty((D, len(theta)))
    cur = lp(theta); acc = 0; rec = 0; recn = 0
    for k in range(B + D):
        prop = theta + rng.normal(0, scales)
        plp = lp(prop)
        if np.log(rng.rand()) < plp - cur:
            theta, cur = prop, plp; acc += 1; rec += 1
        recn += 1
        if k < B and (k + 1) % 200 == 0:
            ar = rec / recn
            if ar < 0.234: scales *= 0.9
            elif ar > 0.50: scales *= 1.1
            rec = 0; recn = 0
        if k >= B: out[k - B] = theta
    return out, acc / (B + D)

def hdi94(s):
    s = np.sort(s); n = len(s); k = int(0.94 * n)
    w = s[k:] - s[:n - k]; i = int(np.argmin(w))
    return float(s[i]), float(s[i + k])

def sd_bf10(beta_samples, sb):
    bm, bs = float(beta_samples.mean()), float(beta_samples.std())
    pri0 = stats.norm.pdf(0.0, 0.0, sb)
    pos0 = stats.norm.pdf(0.0, bm, max(bs, 1e-12))
    return float(pri0 / max(pos0, 1e-300))

# ------------------------------------------------------------- bridge sampling
def bridge_logml(samples, logq, rng, n_prop=None):
    """Bridge-sampling estimate of log int q(phi) dphi.

    Works in the unconstrained parametrisation phi (caller supplies
    samples already transformed, and logq including the Jacobian).
    Proposal: multivariate normal fitted to the posterior samples.
    Meng-Wong optimal bridge, iterated to convergence (log-space).
    """
    n1, dim = samples.shape
    mu = samples.mean(axis=0)
    cov = np.cov(samples.T) + 1e-10 * np.eye(dim)
    if n_prop is None: n_prop = n1
    prop = rng.multivariate_normal(mu, cov, size=n_prop)
    lq_post = np.array([logq(s) for s in samples])          # log q at post draws
    lg_post = stats.multivariate_normal.logpdf(samples, mu, cov)
    lq_prop = np.array([logq(s) for s in prop])             # log q at prop draws
    lg_prop = stats.multivariate_normal.logpdf(prop, mu, cov)
    s1 = n1 / (n1 + n_prop); s2 = n_prop / (n1 + n_prop)
    logr = np.median(lq_post - lg_post)                     # start value
    for _ in range(1000):
        # numerator: E_g[ q / (s1 q + s2 r g) ]  over proposal draws
        num_terms = lq_prop - np.logaddexp(np.log(s1) + lq_prop,
                                           np.log(s2) + logr + lg_prop)
        # denominator: E_post[ g / (s1 q + s2 r g) ] over posterior draws
        den_terms = lg_post - np.logaddexp(np.log(s1) + lq_post,
                                           np.log(s2) + logr + lg_post)
        new_logr = (special.logsumexp(num_terms) - np.log(n_prop)
                    - (special.logsumexp(den_terms) - np.log(n1)))
        if abs(new_logr - logr) < 1e-8: logr = new_logr; break
        logr = new_logr
    return float(logr)

# Prior normalising constants. The MCMC log-posteriors omit them (they
# cancel in Metropolis ratios and in the Savage-Dickey point ratio), but
# a marginal likelihood must include them; omitting them biases log
# BF10 by exactly +log(sb*sqrt(2*pi)) -- caught 2026-10-27 by the
# quadrature cross-check (constant +1.098 log10 offset at sb = 5).
LOG_C_ALPHA = -np.log(PA) - 0.5 * np.log(2 * np.pi)
LOG_C_SIGMA = -np.log(PS) - 0.5 * np.log(np.pi / 2)

def bridge_bf10(xc, y, sb, seed):
    # optional externally supplied M1 samples (see main loop)
    rng = np.random.RandomState(seed + 777)
    log_c_beta = -np.log(sb) - 0.5 * np.log(2 * np.pi)
    # M1 in phi = (a, b, log s). If the caller passes the canonical
    # (released-stream) samples, reuse them so bridge BFs attach to the
    # published chain.
    if bridge_bf10.m1_samples is not None:
        s1samp = bridge_bf10.m1_samples
    else:
        s1samp, _ = run_chain(xc, y, sb, seed, "m1")
    phi1 = np.column_stack([s1samp[:, 0], s1samp[:, 1], np.log(s1samp[:, 2])])
    def lq1(phi):
        a, b, ls = phi; s = np.exp(ls)
        return (log_post_m1((a, b, s), xc, y, sb) + ls
                + LOG_C_ALPHA + log_c_beta + LOG_C_SIGMA)
    logml1 = bridge_logml(phi1, lq1, rng)
    # M0 in phi = (a, log s)
    s0samp, _ = run_chain(xc, y, sb, seed + 55, "m0")
    phi0 = np.column_stack([s0samp[:, 0], np.log(s0samp[:, 1])])
    def lq0(phi):
        a, ls = phi; s = np.exp(ls)
        return (log_post_m0((a, s), xc, y, sb) + ls
                + LOG_C_ALPHA + LOG_C_SIGMA)
    logml0 = bridge_logml(phi0, lq0, rng)
    return logml1 - logml0, logml1, logml0

# ------------------------------------------------------- quadrature truth
def quad_logml(xc, y, sb, model):
    """log p(y|M) by quadrature: alpha integrated analytically given
    (beta, sigma); trapezoid grid over beta (M1) and log sigma."""
    n = len(y); ybar = y.mean(); sxx = np.sum(xc * xc)
    # conditional on (b, s): posterior for a is N(mu_a, va) with
    # va = 1/(n/s^2 + 1/PA^2); marginal factor computed exactly.
    def logml_bs(b, s):
        r = y - b * xc
        va = 1.0 / (n / s**2 + 1.0 / PA**2)
        mu_a = va * (np.sum(r) / s**2)
        ss = np.sum((r - mu_a) ** 2) / s**2 + mu_a**2 / va \
             - mu_a**2 / va  # placeholder, replaced below
        # exact: int N(y; a + b xc, s^2) N(a; 0, PA^2) da
        # = N(y - b xc; 0, s^2 I + PA^2 11') -> use matrix determinant lemma
        z = r - ybar
        # decompose: y - b*xc = ybar*1 + z ; use closed form for equicorr normal
        rss = np.sum((r - r.mean())**2)
        logdet = (n - 1) * np.log(s**2) + np.log(s**2 + n * PA**2)
        quad = rss / s**2 + (r.mean()**2) * n / (s**2 + n * PA**2)
        ll = -0.5 * (n * np.log(2 * np.pi) + logdet + quad)
        return ll
    ls_grid = np.linspace(np.log(0.05), np.log(30.0), 400)
    s_grid = np.exp(ls_grid)
    if model == "m0":
        vals = np.array([logml_bs(0.0, s) - 0.5 * (s / PS) ** 2
                         - np.log(PS) - 0.5 * np.log(np.pi / 2) for s in s_grid])
        # half-normal normalising constant: log pdf = -0.5(s/PS)^2 - log(PS) - 0.5 log(pi/2)
        vals += ls_grid  # Jacobian for log-sigma parametrisation
        m = vals.max()
        return float(m + np.log(np.trapz(np.exp(vals - m), ls_grid)))
    # M1: grid over beta centred on OLS
    _, b_ols, _ = ols(xc, y)
    se_b = np.sqrt(np.mean((y - ybar - b_ols * xc) ** 2) / max(sxx, 1e-12))
    half = max(12 * se_b, 1e-6)
    b_grid = np.linspace(b_ols - half, b_ols + half, 240)
    tot = []
    for s in s_grid:
        ll = np.array([logml_bs(b, s) for b in b_grid])
        lp_b = -0.5 * (b_grid / sb) ** 2 - np.log(sb) - 0.5 * np.log(2 * np.pi)
        lp_s = -0.5 * (s / PS) ** 2 - np.log(PS) - 0.5 * np.log(np.pi / 2)
        vals = ll + lp_b + lp_s
        m = vals.max()
        tot.append(m + np.log(np.trapz(np.exp(vals - m), b_grid)))
    tot = np.array(tot) + ls_grid
    m = tot.max()
    return float(m + np.log(np.trapz(np.exp(tot - m), ls_grid)))

# ------------------------------------------------------------------- R-hat
def split_rhat(chains):
    """Rank-normalised split-Rhat (Vehtari et al. 2021), chains: (m, n)."""
    m, n = chains.shape
    half = n // 2
    split = np.concatenate([chains[:, :half], chains[:, half:2 * half]], axis=0)
    ranks = stats.rankdata(split.ravel())
    z = stats.norm.ppf((ranks - 0.375) / (split.size + 0.25)).reshape(split.shape)
    def rhat_of(v):
        mm, nn = v.shape
        chain_means = v.mean(axis=1)
        bw = nn * chain_means.var(ddof=1)
        w = v.var(axis=1, ddof=1).mean()
        var_hat = (nn - 1) / nn * w + bw / nn
        return np.sqrt(var_hat / w)
    zr = rhat_of(z)
    folded = np.abs(z - np.median(z))
    fr = rhat_of(folded)
    return float(max(zr, fr))

# --------------------------------------------------------------------- main
def main():
    t0 = time.time()
    bridge_bf10.m1_samples = None
    out = {"edges": []}
    # ---- per-edge: replication stats + bridge BF + Rhat (seed i, as released)
    for i, (slug, bs, n, xlo, xhi, sig, d) in enumerate(BENCH):
        rng_edge = np.random.RandomState(i)
        x = rng_edge.uniform(xlo, xhi, n)
        y = 2.0 + bs * (x - x.mean()) + rng_edge.normal(0, sig, n)
        xc = x - x.mean()
        samp, ar = run_chain(xc, y, 5.0, seed=i, model="m1", rng=rng_edge)
        beta = samp[:, 1]
        bm = float(beta.mean()); lo, hi = hdi94(beta)
        pdir = float(np.mean(np.sign(beta) == d))
        sdbf = sd_bf10(beta, 5.0)
        bridge_bf10.m1_samples = samp
        logbf_bridge, lml1, lml0 = bridge_bf10(xc, y, 5.0, seed=i)
        bridge_bf10.m1_samples = None
        q1 = quad_logml(xc, y, 5.0, "m1"); q0 = quad_logml(xc, y, 5.0, "m0")
        quad_logbf = q1 - q0
        chains = []
        for c in range(4):
            cs, _ = run_chain(xc, y, 5.0, seed=1000 * (c + 1) + i, model="m1")
            chains.append(cs[:, 1])
        rhat = split_rhat(np.array(chains))
        dec = ("reject" if sdbf < 1/3 else
               "accept" if (sdbf > 3 and pdir > 0.95) else "inconclusive")
        out["edges"].append(dict(
            slug=slug, beta_star=bs, post_mean=bm, post_sd=float(beta.std()),
            hdi_lo=lo, hdi_hi=hi, p_dir=pdir, accept_rate=ar, decision=dec,
            sd_bf10=min(sdbf, 1e6), sd_saturated=bool(sdbf >= 1e6),
            bridge_log10_bf10=float(logbf_bridge / np.log(10)),
            quad_log10_bf10=float(quad_logbf / np.log(10)),
            bridge_minus_quad_log10=float((logbf_bridge - quad_logbf) / np.log(10)),
            rhat_beta=rhat,
            rel_err=None if bs == 0 else abs(bm - bs) / abs(bs),
            hdi_covers=bool(lo <= bs <= hi)))
        e = out["edges"][-1]
        print(f"{slug:34s} SD_BF={e['sd_bf10']:.3g} bridge_log10BF={e['bridge_log10_bf10']:9.2f} "
              f"quad={e['quad_log10_bf10']:9.2f} diff={e['bridge_minus_quad_log10']:+.3f} Rhat={rhat:.4f}")
    # ---- multi-seed robustness (Savage-Dickey pipeline, 10 master seeds)
    ms = []
    for master in range(10):
        decs, rels, cov, within = {}, [], 0, 0
        null_bf01 = None
        for i, (slug, bs, n, xlo, xhi, sig, d) in enumerate(BENCH):
            x, y = gen_data(bs, n, xlo, xhi, sig, seed=master * 100 + i)
            xc = x - x.mean()
            samp, _ = run_chain(xc, y, 5.0, seed=master * 100 + i, model="m1")
            beta = samp[:, 1]; bm = float(beta.mean())
            lo, hi = hdi94(beta)
            pdir = float(np.mean(np.sign(beta) == d))
            bf = sd_bf10(beta, 5.0)
            dec = ("reject" if bf < 1/3 else
                   "accept" if (bf > 3 and pdir > 0.95) else "inconclusive")
            decs[slug] = dec
            if bs != 0:
                r = abs(bm - bs) / abs(bs); rels.append(r)
                within += r <= 0.09; cov += (lo <= bs <= hi)
            else:
                null_bf01 = 1.0 / bf
        ms.append(dict(master_seed=master, decisions=decs,
                       mean_rel_err=float(np.mean(rels)),
                       n_within_9pct=int(within), n_hdi_covers=int(cov),
                       null_bf01=float(null_bf01)))
        print(f"master {master}: mean_rel_err={np.mean(rels):.4f} within9={within}/18 "
              f"hdi={cov}/18 nullBF01={null_bf01:.1f} pmo={decs['pmo_bottleneck_hits_velocity']} "
              f"n_accept={sum(v=='accept' for v in decs.values())}")
    out["multiseed"] = ms
    pmo_decs = [m["decisions"]["pmo_bottleneck_hits_velocity"] for m in ms]
    out["multiseed_summary"] = dict(
        mean_rel_err_mean=float(np.mean([m["mean_rel_err"] for m in ms])),
        mean_rel_err_range=[float(np.min([m["mean_rel_err"] for m in ms])),
                             float(np.max([m["mean_rel_err"] for m in ms]))],
        within9_range=[int(np.min([m["n_within_9pct"] for m in ms])),
                       int(np.max([m["n_within_9pct"] for m in ms]))],
        hdi_covers_range=[int(np.min([m["n_hdi_covers"] for m in ms])),
                          int(np.max([m["n_hdi_covers"] for m in ms]))],
        null_bf01_range=[float(np.min([m["null_bf01"] for m in ms])),
                         float(np.max([m["null_bf01"] for m in ms]))],
        pmo_decisions=pmo_decs,
        null_decisions=[m["decisions"]["null_demo"] for m in ms])
    # ---- prior sensitivity on the null edge (seed 18, as released ordering)
    i_null = 18
    slug, bs, n, xlo, xhi, sig, d = BENCH[i_null]
    x, y = gen_data(bs, n, xlo, xhi, sig, seed=i_null)
    xc = x - x.mean()
    ps = []
    for sb in [1.0, 2.5, 5.0, 10.0, 25.0]:
        samp, _ = run_chain(xc, y, sb, seed=i_null, model="m1")
        bf_sd = sd_bf10(samp[:, 1], sb)
        lbf, _, _ = bridge_bf10(xc, y, sb, seed=i_null)
        ps.append(dict(sigma_beta=sb, sd_bf01=float(1.0 / bf_sd),
                       bridge_bf01=float(np.exp(-lbf))))
        print(f"sigma_beta={sb:5.1f}  SD BF01={1.0/bf_sd:10.1f}  bridge BF01={np.exp(-lbf):10.1f}")
    out["prior_sensitivity_null"] = ps
    out["wall_time_sec"] = round(time.time() - t0, 1)
    with open("bridge_rhat_multiseed_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote bridge_rhat_multiseed_results.json in", out["wall_time_sec"], "s")

if __name__ == "__main__":
    main()
