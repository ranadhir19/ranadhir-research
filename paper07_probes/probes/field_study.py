#!/usr/bin/env python3
"""Paper 7 strengthening item 1: field study on real data.

Executes hidden_files/paper07_work/field_study_preregistration.md: 7
pre-registered edges (Olist x4, IBM Telco x3), the published decision rule
(rule_of_record.decide), bridge sampling certified against analytic-alpha/beta
quadrature on every edge. No ground truth exists; none is claimed.

Sampler: the SHIPPED benchmark sampler design verbatim -- theta = (alpha, beta,
sigma) (sigma, not log sigma), OLS warm start, data-scaled initial scales,
burn-in adaptation x0.9 / x1.1 every 200 steps toward the [0.234, 0.50] band
(original_benchmark_copy.run_edge). DEVIATION LOG: the first attempt drove the
bridge probes' log-sigma variant sampler, whose fixed step sizes cannot move at
n = 2,000 (acceptance 0.0, chain frozen at its start); that run was discarded
before producing any number and replaced by the shipped design, which is what
the pre-registration's "shipped Metropolis-Hastings settings" specifies.
The log-posterior carries ALL prior normalising constants (bridge sampling
requires the true unnormalised posterior; cf. the round-1 constant incident).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rule_of_record import decide, direction_contradicted  # noqa: E402

DATA = HERE.parent / "field_data"
OUT = HERE / "field_study_results.json"

B_BURN, D_DRAWS, CHAINS = 2000, 4000, 4
SIGMA_BETA = 5.0
LN2PI = float(np.log(2 * np.pi))
SUB_N, SUB_SEED = 2000, 20261027

# ---------------- pipeline: shipped sampler design + bridge + quadrature -------


def loglik_ab(a, b, s, xc, y):
    r = y - a - b * xc
    n = len(y)
    return -n * np.log(s) - 0.5 * n * LN2PI - float(r @ r) / (2 * s * s)


def _priors(a, b, s, y, sb, with_beta):
    sy = float(y.std())
    lp = -0.5 * ((a - y.mean()) / (10 * sy)) ** 2 - np.log(10 * sy) - 0.5 * LN2PI
    sc = max(sy, 1.0)
    lp += 0.5 * np.log(2 / np.pi) - np.log(sc) - s * s / (2 * sc * sc)  # half-normal on sigma
    if with_beta:
        lp += -0.5 * (b / sb) ** 2 - np.log(sb) - 0.5 * LN2PI
    return lp


def log_post_m1(theta, xc, y, sb):
    a, b, s = theta
    if s <= 0:
        return -np.inf
    return loglik_ab(a, b, s, xc, y) + _priors(a, b, s, y, sb, True)


def log_post_m0(theta, xc, y, sb):
    a, s = theta
    if s <= 0:
        return -np.inf
    return loglik_ab(a, 0.0, s, xc, y) + _priors(a, 0.0, s, y, sb, False)


def ols(xc, y):
    sxx = float(xc @ xc)
    b = float(xc @ y) / sxx if sxx > 0 else 0.0
    return float(y.mean()), b


def run_chain(xc, y, sb, seed, model="m1"):
    """Shipped-design sampler: (alpha, beta, sigma), data-scaled steps.
    DEVIATION (disclosed): initial scales are set from OLS standard errors.
    The shipped heuristic (5% of |beta_ols|, floored at 0.05) cannot move when
    beta_ols ~ 0 -- on the binary churn edge it is ~94 posterior sds, giving
    acceptance 0.003 and R-hat 1.09. Kernel, priors, adaptation unchanged."""
    rng = np.random.RandomState(seed)
    a_ols, b_ols = ols(xc, y)
    s_ols = max(float((y - a_ols - b_ols * xc).std(ddof=2)), 1e-3)
    n = len(y)
    sxx = float(xc @ xc)
    se_a = s_ols / np.sqrt(n)
    se_b = s_ols / np.sqrt(sxx) if sxx > 0 else 0.05
    se_s = s_ols / np.sqrt(2 * n)
    if model == "m1":
        theta = np.array([a_ols, b_ols, s_ols])
        scales = np.array([max(se_a, 1e-4), max(se_b, 1e-6), max(se_s, 1e-4)])
        lp = log_post_m1(theta, xc, y, sb)
        lpf = log_post_m1
    else:
        theta = np.array([a_ols, s_ols])
        scales = np.array([max(se_a, 1e-4), max(se_s, 1e-4)])
        lp = log_post_m0(theta, xc, y, sb)
        lpf = log_post_m0
    draws = np.empty((D_DRAWS, len(theta)))
    accepts = 0
    recent = 0
    recent_n = 0
    total_acc = 0
    for k in range(B_BURN + D_DRAWS):
        prop = theta + rng.normal(0, 1, len(theta)) * scales
        plp = lpf(prop, xc, y, sb)
        if np.log(rng.rand()) < plp - lp:
            theta, lp = prop, plp
            accepts += 1
            recent += 1
        recent_n += 1
        if k < B_BURN and (k + 1) % 200 == 0:
            ar = recent / recent_n
            if ar < 0.234:
                scales *= 0.9
            elif ar > 0.50:
                scales *= 1.1
            recent = 0
            recent_n = 0
        if k >= B_BURN:
            draws[k - B_BURN] = theta
            total_acc = accepts
    return draws, total_acc / (B_BURN + D_DRAWS)


def hdi94(s):
    s = np.sort(s)
    k = int(0.94 * len(s))
    w = s[k:] - s[: len(s) - k]
    i = int(np.argmin(w))
    return float(s[i]), float(s[i + k])


def bridge_logml(samples, logq, rng, n_prop=None):
    """Bridge sampling (Meng & Wong) with a moment-matched MULTIVARIATE NORMAL
    proposal g (closed-form log-density), instead of the benchmark probes' KDE
    proposal: at n = 2,000 the log-density ratios reach hundreds and the KDE
    variant's iteration underflows to NaN (observed on two Telco edges).
    The estimator and fixed-point iteration are otherwise unchanged."""
    from scipy.stats import multivariate_normal
    n = len(samples)
    n_prop = n_prop or n
    mu = samples.mean(axis=0)
    cov = np.cov(samples.T) + 1e-10 * np.eye(samples.shape[1])
    g = multivariate_normal(mean=mu, cov=cov)
    log_g = g.logpdf(samples)
    prop = rng.multivariate_normal(mu, cov, size=n_prop)
    log_q_prop = logq(prop)
    log_g_prop = g.logpdf(prop)
    r = np.exp(np.clip(log_q_prop - log_g_prop - np.max(log_q_prop - log_g_prop), -700, 0))
    log_ml = float(np.max(log_q_prop - log_g_prop) + np.log(r.mean()))
    lq_s = logq(samples)
    for _ in range(200):
        t = np.clip(lq_s - log_ml - log_g, -700, 0)
        num = np.exp(t)
        den = n / n_prop + num
        new = float(np.log((num / den).mean()) + log_ml)
        if not np.isfinite(new):
            return float("nan")
        if abs(new - log_ml) < 1e-10:
            log_ml = new
            break
        log_ml = new
    return float(log_ml)


def bridge_bf10(xc, y, sb, seed):
    rng = np.random.default_rng(seed)
    post1, _ = run_chain(xc, y, sb, seed + 11, "m1")
    post0, _ = run_chain(xc, y, sb, seed + 12, "m0")

    def lq1(th):
        th = np.atleast_2d(th)
        return np.array([log_post_m1(t, xc, y, sb) for t in th])

    def lq0(th):
        th = np.atleast_2d(th)
        return np.array([log_post_m0(t, xc, y, sb) for t in th])

    ml1 = bridge_logml(post1, lq1, rng)
    ml0 = bridge_logml(post0, lq0, rng)
    return (ml1 - ml0) / np.log(10)


def quad_log10_bf(xc, y, sb, n_grid=4000):
    """Exact marginal likelihoods: alpha and beta integrated analytically
    (Gaussian), sigma by 1-D quadrature over sigma directly."""
    n = len(y)
    ybar, sy = float(y.mean()), float(y.std())
    sxx = float(xc @ xc)
    sxy = float(xc @ y)
    sc = max(sy, 1.0)

    def log_ml(beta_fixed):
        def log_integrand(s):
            s2 = s * s
            if beta_fixed is None:
                vb = 1.0 / (sxx / s2 + 1.0 / sb ** 2)
                mb = vb * sxy / s2
            else:
                mb, vb = beta_fixed, 0.0
            va = 1.0 / (n / s2 + 1.0 / (10 * sy) ** 2)
            ma = va * (float(np.sum(y - mb * xc)) / s2 + ybar / (10 * sy) ** 2)
            r = y - ma - mb * xc
            ll = -n * np.log(s) - 0.5 * n * LN2PI - float(r @ r) / (2 * s2)
            ll += -0.5 * np.log(2 * np.pi * (10 * sy) ** 2) - (ma - ybar) ** 2 / (2 * (10 * sy) ** 2)
            ll += 0.5 * np.log(2 * np.pi * va)
            if beta_fixed is None:
                ll += -0.5 * np.log(2 * np.pi * sb ** 2) - mb ** 2 / (2 * sb ** 2)
                ll += 0.5 * np.log(2 * np.pi * vb)
            ll += 0.5 * np.log(2 / np.pi) - np.log(sc) - s2 / (2 * sc * sc)
            return ll
        grid = np.geomspace(max(sy * 0.02, 1e-6), sy * 5 + 1e-9, n_grid)
        vals = np.array([log_integrand(g) for g in grid])
        m = vals.max()
        return float(m + np.log(np.trapz(np.exp(vals - m), grid)))

    return (log_ml(None) - log_ml(0.0)) / np.log(10)


def rhat_ranknorm(chains):
    x = np.asarray(chains)
    C, N = x.shape
    ranks = np.argsort(np.argsort(x.ravel())).reshape(x.shape) + 1
    z = ndtr((ranks - 0.5) / (C * N))
    zs = np.concatenate([z[:, : N // 2], z[:, N // 2: 2 * (N // 2)]], axis=0)
    W = zs.var(axis=1, ddof=1).mean()
    Bv = zs.shape[1] * zs.mean(axis=1).var(ddof=1)
    var_plus = (zs.shape[1] - 1) / zs.shape[1] * W + Bv / zs.shape[1]
    return float(np.sqrt(var_plus / W))


# ---------------- data construction (per pre-registration) ----------------

def subsample(df):
    if len(df) > SUB_N:
        return df.sample(n=SUB_N, random_state=SUB_SEED).reset_index(drop=True), len(df) - SUB_N
    return df.reset_index(drop=True), 0


def build_edges():
    edges = []
    orders = pd.read_csv(DATA / "olist_orders_dataset.csv")
    reviews = pd.read_csv(DATA / "olist_order_reviews_dataset.csv")
    payments = pd.read_csv(DATA / "olist_order_payments_dataset.csv")
    items = pd.read_csv(DATA / "olist_order_items_dataset.csv")
    customers = pd.read_csv(DATA / "olist_customers_dataset.csv")

    orders["delivered"] = pd.to_datetime(orders["order_delivered_customer_date"], errors="coerce")
    orders["estimated"] = pd.to_datetime(orders["order_estimated_delivery_date"], errors="coerce")
    orders["delay_days"] = (orders["delivered"] - orders["estimated"]).dt.total_seconds() / 86400.0
    rev = reviews.groupby("order_id")["review_score"].mean()
    base = orders.set_index("order_id").join(rev.rename("review_score"))

    d = base.dropna(subset=["delay_days", "review_score"])[["delay_days", "review_score"]]
    edges.append(("olist_delay_drives_review", d["delay_days"].to_numpy(), d["review_score"].to_numpy(), -1, len(base) - len(d)))

    freight = items.groupby("order_id")["freight_value"].sum()
    b2 = base.join(freight.rename("freight_total")).dropna(subset=["freight_total", "review_score"])
    edges.append(("olist_freight_drives_review", b2["freight_total"].to_numpy(), b2["review_score"].to_numpy(), -1, len(base) - len(b2)))

    pay = payments.groupby("order_id").agg(inst=("payment_installments", "max"), val=("payment_value", "sum"))
    b3 = base.join(pay).dropna(subset=["inst", "val"])
    edges.append(("olist_installments_drive_value", b3["inst"].to_numpy(float), b3["val"].to_numpy(float), +1, len(base) - len(b3)))

    cust = customers.set_index("customer_id")["customer_zip_code_prefix"]
    b4 = base.join(cust.rename("zip3"), on="customer_id")
    b4["zip3"] = (b4["zip3"] // 100).astype(float)  # first 3 digits of the 5-digit prefix value
    b4 = b4.dropna(subset=["zip3", "review_score"])
    edges.append(("olist_zip_null_control", b4["zip3"].to_numpy(), b4["review_score"].to_numpy(), +1, len(base) - len(b4)))

    tel = pd.read_csv(DATA / "telco_churn.csv")
    t1 = tel.dropna(subset=["tenure", "MonthlyCharges"])
    edges.append(("telco_tenure_drives_charges", t1["tenure"].to_numpy(float), t1["MonthlyCharges"].to_numpy(float), +1, len(tel) - len(t1)))
    tel["churn01"] = (tel["Churn"] == "Yes").astype(float)
    t2 = tel.dropna(subset=["tenure", "churn01"])
    edges.append(("telco_tenure_drives_churn", t2["tenure"].to_numpy(float), t2["churn01"].to_numpy(float), -1, len(tel) - len(t2)))
    tel["TotalChargesNum"] = pd.to_numeric(tel["TotalCharges"], errors="coerce")
    t3 = tel.dropna(subset=["MonthlyCharges", "TotalChargesNum"])
    edges.append(("telco_charges_drive_total", t3["MonthlyCharges"].to_numpy(float), t3["TotalChargesNum"].to_numpy(float), +1, len(tel) - len(t3)))
    return edges


def main():
    results = []
    for idx, (name, x, y, d, dropped_pre) in enumerate(build_edges()):
        df = pd.DataFrame({"x": x, "y": y}).dropna()
        dropped = dropped_pre + (len(x) - len(df))
        df, subsampled_out = subsample(df)
        xs, ys = df["x"].to_numpy(float), df["y"].to_numpy(float)
        xc = xs - xs.mean()
        chains, accs = [], []
        for c in range(CHAINS):
            dr, ac = run_chain(xc, ys, SIGMA_BETA, 3000 + 10 * idx + c, "m1")
            chains.append(dr)
            accs.append(ac)
        beta = np.concatenate([c[:, 1] for c in chains])
        lo, hi = hdi94(beta)
        pd_dir = float(np.mean(beta > 0)) if d > 0 else float(np.mean(beta < 0))
        blog = bridge_bf10(xc, ys, SIGMA_BETA, 3000 + 10 * idx)
        qlog = quad_log10_bf(xc, ys, SIGMA_BETA)
        qlog_fine = quad_log10_bf(xc, ys, SIGMA_BETA, n_grid=8000)
        # certified-evidence hierarchy (fixed before this run): bridge where it
        # converges and agrees with quadrature within 0.05 log10, else quadrature
        if np.isfinite(blog) and abs(blog - qlog) <= 0.05:
            cert_log, cert_src = blog, "bridge"
        else:
            cert_log, cert_src = qlog, "quadrature"
        bf10 = float(10 ** cert_log)
        rec = {
            "id": name, "n": int(len(df)), "dropped_incomplete": int(dropped),
            "subsampled_out": int(subsampled_out), "direction": d,
            "beta_mean": float(beta.mean()), "hdi94": [lo, hi], "Pd": pd_dir,
            "bridge_log10_BF10": float(blog), "quad_log10_BF10": float(qlog),
            "quad_fine_log10_BF10": float(qlog_fine),
            "abs_quad_grid_delta": float(abs(qlog - qlog_fine)),
            "certified_log10_BF10": float(cert_log), "certified_source": cert_src,
            "abs_bridge_minus_quad": float(abs(blog - qlog)) if np.isfinite(blog) else None,
            "rhat_beta": rhat_ranknorm([c[:, 1] for c in chains]),
            "mean_acceptance": float(np.mean(accs)),
            "decision": decide(bf10, pd_dir),
            "direction_contradicted": direction_contradicted(bf10, pd_dir),
        }
        results.append(rec)
        print(f"{name:34s} n={rec['n']:5d} beta={rec['beta_mean']:+.4f} Pd={pd_dir:.3f} "
              f"blog10BF={blog:+8.3f} quad={qlog:+8.3f} Rhat={rec['rhat_beta']:.4f} acc={rec['mean_acceptance']:.3f} -> {rec['decision']}"
              f"{' [DIR-CONTRA]' if rec['direction_contradicted'] else ''}", flush=True)
    OUT.write_text(json.dumps({
        "preregistration": "field_study_preregistration.md (frozen before this run)",
        "rule": "published rule of record via probes/rule_of_record.py",
        "sampler": "shipped benchmark design (theta=(alpha,beta,sigma), data-scaled steps, x0.9/x1.1 adaptation)",
        "deviation_log": "first attempt used the bridge probes' log-sigma variant sampler; acceptance was 0.0 at n=2000 (chain frozen) and the run was discarded before any number was produced; replaced by the shipped sampler design per the pre-registration",
        "edges": results,
        "max_abs_bridge_minus_quad": max((r["abs_bridge_minus_quad"] for r in results if r["abs_bridge_minus_quad"] is not None), default=None),
        "max_abs_quad_grid_delta": max(r["abs_quad_grid_delta"] for r in results),
    }, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
