#!/usr/bin/env python3
"""Paper 7 upgrade: field ledger run, exactly per upgrade_preregistration.md.

21 edges: stage-1 (bivariate, same deterministic protocol, empty covariates),
stage-2 (adjusted), ledger rule at w in {0.1, 0.5, 0.9}. Also certifies the
adjusted computation with an empty covariate set against the EXISTING
bivariate quadrature (probes/field_study.py: quad_log10_bf) on the 7
original edges (same standardised inputs).
Outputs: field_upgrade_results.json (every number the manuscript prints).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(WORK / "probes"))
from adjusted_core import fit_adjusted, p_claim, ledger_prefix  # noqa: E402
from rule_of_record import decide  # noqa: E402
from field_study import quad_log10_bf  # noqa: E402  (existing pipeline)

DATA = WORK / "field_data"
SUB_N, SUB_SEED = 2000, 20261027


def complete_sample(df, cols):
    d = df.dropna(subset=cols).reset_index(drop=True)
    dropped = len(df) - len(d)
    if len(d) > SUB_N:
        d = d.sample(n=SUB_N, random_state=SUB_SEED).reset_index(drop=True)
    return d, dropped


def olist_frame():
    orders = pd.read_csv(DATA / "olist_orders_dataset.csv")
    reviews = pd.read_csv(DATA / "olist_order_reviews_dataset.csv")
    payments = pd.read_csv(DATA / "olist_order_payments_dataset.csv")
    items = pd.read_csv(DATA / "olist_order_items_dataset.csv")
    customers = pd.read_csv(DATA / "olist_customers_dataset.csv")
    orders["delivered"] = pd.to_datetime(orders["order_delivered_customer_date"], errors="coerce")
    orders["estimated"] = pd.to_datetime(orders["order_estimated_delivery_date"], errors="coerce")
    orders["delay_days"] = (orders["delivered"] - orders["estimated"]).dt.total_seconds() / 86400.0
    rev = reviews.groupby("order_id")["review_score"].mean().rename("review_score")
    fr = items.groupby("order_id").agg(freight_total=("freight_value", "sum"),
                                       items_value=("price", "sum"),
                                       n_items=("order_item_id", "count"))
    pay = payments.groupby("order_id").agg(inst=("payment_installments", "max"),
                                           order_value=("payment_value", "sum"))
    cust = customers.set_index("customer_id")
    states = sorted(cust["customer_state"].dropna().unique())
    state_rank = {s: i + 1 for i, s in enumerate(states)}
    base = orders.set_index("order_id").join(rev).join(fr).join(pay)
    base = base.join(cust["customer_zip_code_prefix"].rename("zip_prefix"), on="customer_id")
    base = base.join(cust["customer_state"].rename("state"), on="customer_id")
    base["zip3"] = (base["zip_prefix"] // 100).astype(float)
    base["state_code"] = base["state"].map(state_rank).astype(float)
    return base.reset_index()


def telco_frame():
    tel = pd.read_csv(DATA / "telco_churn.csv")
    tel["churn01"] = (tel["Churn"] == "Yes").astype(float)
    tel["Partner01"] = (tel["Partner"] == "Yes").astype(float)
    tel["Dependents01"] = (tel["Dependents"] == "Yes").astype(float)
    tel["Paperless01"] = (tel["PaperlessBilling"] == "Yes").astype(float)
    tel["M2M01"] = (tel["Contract"] == "Month-to-month").astype(float)
    tel["TotalChargesNum"] = pd.to_numeric(tel["TotalCharges"], errors="coerce")
    tel["SeniorCitizen"] = tel["SeniorCitizen"].astype(float)
    tel["tenure"] = tel["tenure"].astype(float)
    return tel


def bike_frame():
    import zipfile
    with zipfile.ZipFile(WORK / "upgrade_data" / "bike_sharing.zip") as z:
        with z.open("day.csv") as f:
            return pd.read_csv(f)


# (id, frame key, x col, y col, direction, [covariate cols])
def edge_specs(O, T, B):
    return [
        ("olist_delay_drives_review", O, "delay_days", "review_score", -1, ["freight_total", "items_value", "n_items"]),
        ("olist_freight_drives_review", O, "freight_total", "review_score", -1, ["delay_days", "items_value", "n_items"]),
        ("olist_installments_drive_value", O, "inst", "order_value", +1, ["n_items", "freight_total"]),
        ("olist_zip_null_control", O, "zip3", "review_score", +1, ["delay_days", "freight_total"]),
        ("telco_tenure_drives_charges", T, "tenure", "MonthlyCharges", +1, ["SeniorCitizen", "Partner01", "M2M01"]),
        ("telco_tenure_drives_churn", T, "tenure", "churn01", -1, ["MonthlyCharges", "SeniorCitizen", "M2M01"]),
        ("telco_charges_drive_total", T, "MonthlyCharges", "TotalChargesNum", +1, ["tenure", "SeniorCitizen"]),
        ("olist_nitems_drive_freight", O, "n_items", "freight_total", +1, ["items_value", "delay_days"]),
        ("olist_freight_drives_delay", O, "freight_total", "delay_days", +1, ["items_value", "n_items"]),
        ("olist_review_drives_freight", O, "review_score", "freight_total", +1, ["items_value", "n_items"]),
        ("olist_statecode_drives_review", O, "state_code", "review_score", +1, ["delay_days", "freight_total"]),
        ("olist_nitems_drive_review", O, "n_items", "review_score", -1, ["delay_days", "freight_total", "items_value"]),
        ("telco_senior_drives_churn", T, "SeniorCitizen", "churn01", +1, ["tenure", "MonthlyCharges", "M2M01"]),
        ("telco_partner_drives_tenure", T, "Partner01", "tenure", +1, ["SeniorCitizen", "Dependents01"]),
        ("telco_charges_drive_churn", T, "MonthlyCharges", "churn01", +1, ["tenure", "SeniorCitizen", "M2M01"]),
        ("telco_paperless_drives_churn", T, "Paperless01", "churn01", +1, ["tenure", "MonthlyCharges", "M2M01"]),
        ("bike_temp_drives_cnt", B, "temp", "cnt", +1, ["hum", "windspeed", "workingday", "yr"]),
        ("bike_hum_drives_cnt", B, "hum", "cnt", -1, ["temp", "windspeed", "workingday", "yr"]),
        ("bike_windspeed_drives_cnt", B, "windspeed", "cnt", -1, ["temp", "hum", "workingday", "yr"]),
        ("bike_workingday_drives_cnt", B, "workingday", "cnt", +1, ["temp", "hum", "windspeed", "yr"]),
        ("bike_yr_drives_cnt", B, "yr", "cnt", +1, ["temp", "hum", "windspeed", "workingday"]),
    ]


def main():
    O, T, B = olist_frame(), telco_frame(), bike_frame()
    published = {e["id"]: e for e in json.loads(
        (WORK / "probes" / "field_study_results.json").read_text())["edges"]}
    rows = []
    cert = []
    for eid, frame, xc, yc, d, covs in edge_specs(O, T, B):
        cols = [xc, yc] + covs
        dfr, dropped = complete_sample(frame, cols)
        x = dfr[xc].to_numpy(float); y = dfr[yc].to_numpy(float)
        C = [dfr[c].to_numpy(float) for c in covs]
        s1 = fit_adjusted(y, x, [], d)
        s2 = fit_adjusted(y, x, C, d)
        dec1 = decide(s1["BF10"], s1["Pd"])
        dec2 = decide(s2["BF10"], s2["Pd"])
        stage2_rat = (dec1 == "accept" and dec2 == "accept")
        pc = {}
        for w in (0.1, 0.5, 0.9):
            p, _ = p_claim(s2["BF10"], s2["Pd"], w, log_bf10=s2["log_BF10"])
            pc[str(w)] = p
        row = {
            "id": eid, "x": xc, "y": yc, "direction": d, "covariates": covs,
            "n": int(len(dfr)), "dropped_incomplete": int(dropped),
            "stage1_log10_BF10": s1["log10_BF10"], "stage1_Pd": s1["Pd"],
            "stage1_beta_mean_std": s1["beta_mean_std"], "stage1_decision": dec1,
            "adj_log10_BF10": s2["log10_BF10"], "adj_Pd": s2["Pd"],
            "adj_beta_mean_std": s2["beta_mean_std"], "stage2_decision": dec2,
            "stage2_ratified": bool(stage2_rat),
            "p_claim": pc,
            "published_stage1_decision": published.get(eid, {}).get("decision"),
        }
        rows.append(row)
        print(f"{eid:34s} n={row['n']:5d} S1 logBF={s1['log10_BF10']:+8.3f} {dec1:12s} "
              f"ADJ logBF={s2['log10_BF10']:+8.3f} Pd={s2['Pd']:.3f} {dec2:12s} "
              f"rat2={stage2_rat} p_claim(0.5)={pc['0.5']:.4f}", flush=True)
        if eid in published:  # certification on the 7 original edges
            zx = (x - x.mean()) / x.std(); zy = (y - y.mean()) / y.std()
            q = quad_log10_bf(zx - zx.mean(), zy, 5.0)
            cert.append({"id": eid, "adjusted_empty_log10_BF10": s1["log10_BF10"],
                         "existing_quad_log10_BF10": float(q),
                         "abs_discrepancy": float(abs(s1["log10_BF10"] - q))})
    ledger = {}
    for w in ("0.1", "0.5", "0.9"):
        rat_idx = [i for i, r in enumerate(rows) if r["stage2_ratified"]]
        sub = [{"p_claim": rows[i]["p_claim"][w]} for i in rat_idx]
        keep_local = ledger_prefix(sub)
        keep_ids = sorted(rows[rat_idx[j]]["id"] for j in keep_local)
        ledger[w] = {"stage2_ratified_count": len(rat_idx),
                     "ledger_ratified_count": len(keep_ids),
                     "ledger_ratified_ids": keep_ids}
        for j, i in enumerate(rat_idx):
            rows[i].setdefault("ledger_ratified", {})[w] = bool(j in keep_local)
    out = {
        "preregistration": "upgrade_preregistration.md (frozen before this run)",
        "protocol": "standardised deterministic adjusted computation (adjusted_core.py); rule of record at both stages; ledger gamma=0.05",
        "certification_empty_covariate_vs_existing_quadrature": {
            "edges": cert,
            "max_abs_discrepancy": max(c["abs_discrepancy"] for c in cert),
            "criterion": "<= 0.01 log10 units on each of the 7 original edges",
        },
        "edges": rows,
        "ledger_summary": ledger,
        "counts": {
            "total_edges": len(rows),
            "stage1_accept": sum(r["stage1_decision"] == "accept" for r in rows),
            "stage1_reject": sum(r["stage1_decision"] == "reject" for r in rows),
            "stage1_inconclusive": sum(r["stage1_decision"] == "inconclusive" for r in rows),
            "stage2_ratified": sum(r["stage2_ratified"] for r in rows),
            "changed_stage1_to_stage2": [r["id"] for r in rows
                                         if (r["stage1_decision"] == "accept") != r["stage2_ratified"]],
        },
    }
    (HERE / "field_upgrade_results.json").write_text(json.dumps(out, indent=2))
    print("certification max |Δ| =", out["certification_empty_covariate_vs_existing_quadrature"]["max_abs_discrepancy"])
    print("ledger:", json.dumps(ledger, indent=1))
    print("wrote", HERE / "field_upgrade_results.json")


if __name__ == "__main__":
    main()
