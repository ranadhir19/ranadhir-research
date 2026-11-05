#!/usr/bin/env python3
"""R2 tester probe 1: INDEPENDENT field-study re-derivation.
Written by the round-2 tester; does NOT import builder field_study code.
For all 7 pre-registered edges: rebuild (x,y) from raw CSVs per the
pre-registration text, take the frozen n=2000 subsample (seed 20261027),
compute OLS beta, and compute log10 BF10 / posterior beta mean / Pd / 94% HDI
by the tester's own quadrature: alpha,beta integrated analytically given
sigma (2x2 Gaussian), sigma integrated on a fine log-grid with the
half-normal prior. Three prior sets are evaluated:
  A field-code priors : alpha~N(ybar,(10*sd_y)^2), sigma~HN(max(sd_y,1)), beta~N(0,5^2)
  B prereg-literal    : alpha~N(ybar,10^2),       sigma~HN(max(sd_y,1)), beta~N(0,5^2)
  C manuscript Eq.(2) : alpha~N(0,10^2),          sigma~HN(5),           beta~N(0,5^2)
Decisions use the published rule re-implemented here.
Outputs JSON + prints a comparison vs builder field_study_results.json.
"""
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "field_data"
SUB_N, SUB_SEED, SB = 2000, 20261027, 5.0
LN10 = np.log(10.0)

def decide(bf, pd_):
    if bf >= 10 and pd_ >= 0.95: return "accept"
    if 3 <= bf < 10 and pd_ >= 0.90: return "accept"
    if bf <= 1/3: return "reject"
    return "inconclusive"

def quad_all(x, y, a_mean, a_sd, s_scale, with_beta):
    """Return log marginal likelihood (alpha[,beta] integrated, sigma quad),
    plus posterior moments of beta (mixture over sigma posterior)."""
    n = len(y); xc = x - x.mean()
    sxx = float(xc @ xc); sxy = float(xc @ y); sy_sum = float(y.sum())
    # sigma grid: log-spaced around residual scale
    resid_sd = float(np.std(y - (y.mean() + (sxy/sxx if sxx>0 else 0)*xc)))
    grid = np.geomspace(max(resid_sd*0.05, 1e-8), max(resid_sd*8, 1e-6), 3000)
    logw = np.empty_like(grid); mb_arr = np.empty_like(grid); vb_arr = np.empty_like(grid)
    for i, s in enumerate(grid):
        s2 = s*s
        # joint Gaussian integral over (alpha,beta) or alpha only
        if with_beta:
            # precision matrix for (a,b): diag-ish with cross term 0 because sum xc = 0
            pa = 1.0/a_sd**2 + n/s2
            pb = 1.0/SB**2 + sxx/s2
            va, vb = 1.0/pa, 1.0/pb
            ma = va*(a_mean/a_sd**2 + sy_sum/s2)
            mb = vb*(sxy/s2)  # beta prior mean 0
            r = y - ma - mb*xc
            ll = -n*np.log(s) - 0.5*n*np.log(2*np.pi) - float(r@r)/(2*s2)
            ll += -0.5*np.log(2*np.pi*a_sd**2) - (ma-a_mean)**2/(2*a_sd**2) + 0.5*np.log(2*np.pi*va)
            ll += -0.5*np.log(2*np.pi*SB**2) - mb**2/(2*SB**2) + 0.5*np.log(2*np.pi*vb)
        else:
            pa = 1.0/a_sd**2 + n/s2
            va = 1.0/pa; ma = va*(a_mean/a_sd**2 + sy_sum/s2)
            mb, vb = 0.0, 0.0
            r = y - ma
            ll = -n*np.log(s) - 0.5*n*np.log(2*np.pi) - float(r@r)/(2*s2)
            ll += -0.5*np.log(2*np.pi*a_sd**2) - (ma-a_mean)**2/(2*a_sd**2) + 0.5*np.log(2*np.pi*va)
        # half-normal prior on sigma (normalising const included)
        ll += 0.5*np.log(2/np.pi) - np.log(s_scale) - s2/(2*s_scale**2)
        logw[i] = ll; mb_arr[i] = mb; vb_arr[i] = vb
    m = logw.max()
    w_unnorm = np.exp(logw - m)
    logml = m + np.log(np.trapz(w_unnorm, grid))
    post_w = w_unnorm * np.trapz(np.ones_like(grid), grid)  # placeholder, use normalised weights below
    # posterior weights over sigma (density in s, trapezoid weights)
    tw = np.gradient(grid)  # approx cell widths (log grid: gradient is fine for weighting check)
    # use trapezoid-consistent weights:
    tw = np.zeros_like(grid); tw[1:-1] = (grid[2:]-grid[:-2])/2; tw[0]=(grid[1]-grid[0])/2; tw[-1]=(grid[-1]-grid[-2])/2
    pw = w_unnorm * tw; pw /= pw.sum()
    if with_beta:
        bmean = float((pw*mb_arr).sum())
        # Pd and HDI from the sigma-mixture CDF on a beta grid
        lo_b, hi_b = bmean - 8*np.sqrt(float((pw*(vb_arr+mb_arr**2)).sum())-bmean**2), bmean + 8*np.sqrt(float((pw*(vb_arr+mb_arr**2)).sum())-bmean**2)
        bg = np.linspace(lo_b, hi_b, 4001)
        cdf = (pw[:,None]*norm.cdf((bg[None,:]-mb_arr[:,None])/np.sqrt(vb_arr)[:,None])).sum(axis=0)
        pd_pos = 1.0 - float(np.interp(0.0, bg, cdf))
        # 94% HDI of a unimodal mixture: shortest window with mass .94 via CDF
        q = lambda p: float(np.interp(p, cdf, bg))
        best=None
        for a in np.linspace(0,0.06,121):
            wdt = q(a+0.94)-q(a)
            if best is None or wdt<best[0]: best=(wdt,q(a),q(a+0.94))
        return logml, bmean, pd_pos, (best[1], best[2])
    return logml, None, None, None

def analyse(x, y, d):
    sy = float(y.std()); ybar = float(y.mean())
    sets = {
        "A_field_code":   (ybar, 10*sy, max(sy,1.0)),
        "B_prereg_literal":(ybar, 10.0, max(sy,1.0)),
        "C_manuscript_eq2":(0.0, 10.0, 5.0),
    }
    out={}
    for name,(am,asd,ssc) in sets.items():
        lm1,bm,pdpos,hdi = quad_all(x,y,am,asd,ssc,True)
        lm0,_,_,_ = quad_all(x,y,am,asd,ssc,False)
        logbf = (lm1-lm0)/LN10
        pd_ = pdpos if d>0 else 1.0-pdpos
        out[name]=dict(log10BF10=logbf, beta_mean=bm, Pd=pd_, hdi94=list(hdi),
                       decision=decide(10**logbf, pd_))
    xc = x-x.mean(); bols = float(xc@y)/float(xc@xc)
    out["ols_beta"]=bols; out["sd_y"]=sy
    return out

def build():
    orders = pd.read_csv(DATA/"olist_orders_dataset.csv")
    reviews = pd.read_csv(DATA/"olist_order_reviews_dataset.csv")
    payments = pd.read_csv(DATA/"olist_order_payments_dataset.csv")
    items = pd.read_csv(DATA/"olist_order_items_dataset.csv")
    customers = pd.read_csv(DATA/"olist_customers_dataset.csv")
    rows_info = {f: len(pd.read_csv(DATA/f)) for f in
                 ["olist_orders_dataset.csv","olist_order_reviews_dataset.csv",
                  "olist_order_payments_dataset.csv","olist_order_items_dataset.csv",
                  "olist_customers_dataset.csv","telco_churn.csv"]}
    hashes = {f: hashlib.sha256((DATA/f).read_bytes()).hexdigest() for f in rows_info}
    orders["delay_days"]=(pd.to_datetime(orders["order_delivered_customer_date"],errors="coerce")
                          -pd.to_datetime(orders["order_estimated_delivery_date"],errors="coerce")).dt.total_seconds()/86400.0
    rev = reviews.groupby("order_id")["review_score"].mean()
    base = orders.set_index("order_id").join(rev.rename("review_score"))
    E=[]
    d = base.dropna(subset=["delay_days","review_score"])
    E.append(("olist_delay_drives_review", d["delay_days"].to_numpy(float), d["review_score"].to_numpy(float), -1))
    fr = items.groupby("order_id")["freight_value"].sum()
    b2 = base.join(fr.rename("freight_total")).dropna(subset=["freight_total","review_score"])
    E.append(("olist_freight_drives_review", b2["freight_total"].to_numpy(float), b2["review_score"].to_numpy(float), -1))
    pay = payments.groupby("order_id").agg(inst=("payment_installments","max"), val=("payment_value","sum"))
    b3 = base.join(pay).dropna(subset=["inst","val"])
    E.append(("olist_installments_drive_value", b3["inst"].to_numpy(float), b3["val"].to_numpy(float), +1))
    cust = customers.set_index("customer_id")["customer_zip_code_prefix"]
    b4 = base.join(cust.rename("zip"), on="customer_id")
    b4["zip3"]=(b4["zip"]//100).astype(float)
    b4=b4.dropna(subset=["zip3","review_score"])
    E.append(("olist_zip_null_control", b4["zip3"].to_numpy(float), b4["review_score"].to_numpy(float), +1))
    tel = pd.read_csv(DATA/"telco_churn.csv")
    E.append(("telco_tenure_drives_charges", tel["tenure"].to_numpy(float), tel["MonthlyCharges"].to_numpy(float), +1))
    churn=(tel["Churn"]=="Yes").astype(float)
    E.append(("telco_tenure_drives_churn", tel["tenure"].to_numpy(float), churn.to_numpy(float), -1))
    tot = pd.to_numeric(tel["TotalCharges"], errors="coerce")
    m = tot.notna()
    E.append(("telco_charges_drive_total", tel.loc[m,"MonthlyCharges"].to_numpy(float), tot[m].to_numpy(float), +1))
    zipstats = dict(zip_prefix_min=float(customers["customer_zip_code_prefix"].min()),
                    zip_prefix_max=float(customers["customer_zip_code_prefix"].max()),
                    zip_prefix_nunique=int(customers["customer_zip_code_prefix"].nunique()))
    return E, rows_info, hashes, zipstats

def main():
    E, rows_info, hashes, zipstats = build()
    builder = {e["id"]: e for e in json.loads((BASE/"probes"/"field_study_results.json").read_text())["edges"]}
    out = dict(rows=rows_info, sha256=hashes, zipstats=zipstats, edges={})
    for name,x,y,d in E:
        df = pd.DataFrame({"x":x,"y":y}).dropna()
        nfull=len(df)
        if nfull>SUB_N: df=df.sample(n=SUB_N, random_state=SUB_SEED)
        xs,ys = df["x"].to_numpy(float), df["y"].to_numpy(float)
        r = analyse(xs,ys,d)
        b = builder[name]
        r["n_full_complete"]=int(nfull); r["n_used"]=int(len(df))
        r["builder"]=dict(beta_mean=b["beta_mean"], hdi94=b["hdi94"], Pd=b["Pd"],
                          bridge=b["bridge_log10_BF10"], quad=b["quad_log10_BF10"],
                          rhat=b["rhat_beta"], decision=b["decision"])
        r["testerA_minus_builderQuad"]=r["A_field_code"]["log10BF10"]-b["quad_log10_BF10"]
        r["testerA_beta_minus_builder"]=r["A_field_code"]["beta_mean"]-b["beta_mean"]
        out["edges"][name]=r
        print(f"{name:34s} n={len(df)} OLS={r['ols_beta']:+.6f} "
              f"A: logBF={r['A_field_code']['log10BF10']:+.3f} beta={r['A_field_code']['beta_mean']:+.5f} Pd={r['A_field_code']['Pd']:.3f} {r['A_field_code']['decision']} | "
              f"B: {r['B_prereg_literal']['log10BF10']:+.3f}/{r['B_prereg_literal']['decision']} "
              f"C: {r['C_manuscript_eq2']['log10BF10']:+.3f}/{r['C_manuscript_eq2']['decision']} | "
              f"builderQuad={b['quad_log10_BF10']:+.3f} dA-quad={r['testerA_minus_builderQuad']:+.4f}")
    (Path(__file__).parent/"probe_r2_field_results.json").write_text(json.dumps(out,indent=2))
    print("wrote probe_r2_field_results.json")

if __name__=="__main__":
    main()
