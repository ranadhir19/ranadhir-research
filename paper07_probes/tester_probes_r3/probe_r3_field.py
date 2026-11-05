#!/usr/bin/env python3
"""R3 field rebuild: tester's OWN construction from the raw CSVs, following
only the upgrade pre-registration text. No builder code imported (fit/decide
are the tester's own my_core). Rebuilds ALL 21 edges; compares every printed
stage-1/adjusted log10BF, Pd, decision, ratification and p_claim(w=0.5) to
field_upgrade_results.json and to the manuscript tables (entered in the
comparison script from the .tex). Outputs probe_r3_field_results.json.
"""
import json, sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from my_core import fit, decide, p_claim_from_logbf, ledger_keep

WORK = Path(__file__).parent.parent
FD = WORK/"field_data"

def build_frames():
    orders=pd.read_csv(FD/"olist_orders_dataset.csv")
    reviews=pd.read_csv(FD/"olist_order_reviews_dataset.csv")
    payments=pd.read_csv(FD/"olist_order_payments_dataset.csv")
    items=pd.read_csv(FD/"olist_order_items_dataset.csv")
    customers=pd.read_csv(FD/"olist_customers_dataset.csv")
    dl=pd.to_datetime(orders["order_delivered_customer_date"],errors="coerce")
    es=pd.to_datetime(orders["order_estimated_delivery_date"],errors="coerce")
    orders["delay_days"]=(dl-es).dt.total_seconds()/86400.0
    rev=reviews.groupby("order_id")["review_score"].mean()
    agg=items.groupby("order_id").agg(freight_total=("freight_value","sum"),
                                      items_value=("price","sum"),
                                      n_items=("order_item_id","count"))
    pay=payments.groupby("order_id").agg(inst=("payment_installments","max"),
                                         order_value=("payment_value","sum"))
    base=orders.set_index("order_id").join(rev.rename("review_score")).join(agg).join(pay)
    cust=customers.set_index("customer_id")
    base=base.join(cust["customer_zip_code_prefix"].rename("zip_prefix"),on="customer_id")
    base=base.join(cust["customer_state"].rename("state"),on="customer_id")
    base["zip3"]=(base["zip_prefix"]//100).astype(float)
    ranks={s:i+1 for i,s in enumerate(sorted(cust["customer_state"].dropna().unique()))}
    base["state_code"]=base["state"].map(ranks).astype(float)
    O=base.reset_index()
    T=pd.read_csv(FD/"telco_churn.csv")
    T["churn01"]=(T["Churn"]=="Yes").astype(float)
    T["Partner01"]=(T["Partner"]=="Yes").astype(float)
    T["Dependents01"]=(T["Dependents"]=="Yes").astype(float)
    T["Paperless01"]=(T["PaperlessBilling"]=="Yes").astype(float)
    T["M2M01"]=(T["Contract"]=="Month-to-month").astype(float)
    T["TotalChargesNum"]=pd.to_numeric(T["TotalCharges"],errors="coerce")
    T["SeniorCitizen"]=T["SeniorCitizen"].astype(float); T["tenure"]=T["tenure"].astype(float)
    with zipfile.ZipFile(WORK/"upgrade_data"/"bike_sharing.zip") as z:
        with z.open("day.csv") as f: B=pd.read_csv(f)
    return O,T,B

def edge_specs(O,T,B):
    return [
      ("olist_delay_drives_review",O,"delay_days","review_score",-1,["freight_total","items_value","n_items"]),
      ("olist_freight_drives_review",O,"freight_total","review_score",-1,["delay_days","items_value","n_items"]),
      ("olist_installments_drive_value",O,"inst","order_value",1,["n_items","freight_total"]),
      ("olist_zip_null_control",O,"zip3","review_score",1,["delay_days","freight_total"]),
      ("telco_tenure_drives_charges",T,"tenure","MonthlyCharges",1,["SeniorCitizen","Partner01","M2M01"]),
      ("telco_tenure_drives_churn",T,"tenure","churn01",-1,["MonthlyCharges","SeniorCitizen","M2M01"]),
      ("telco_charges_drive_total",T,"MonthlyCharges","TotalChargesNum",1,["tenure","SeniorCitizen"]),
      ("olist_nitems_drive_freight",O,"n_items","freight_total",1,["items_value","delay_days"]),
      ("olist_freight_drives_delay",O,"freight_total","delay_days",1,["items_value","n_items"]),
      ("olist_review_drives_freight",O,"review_score","freight_total",1,["items_value","n_items"]),
      ("olist_statecode_drives_review",O,"state_code","review_score",1,["delay_days","freight_total"]),
      ("olist_nitems_drive_review",O,"n_items","review_score",-1,["delay_days","freight_total","items_value"]),
      ("telco_senior_drives_churn",T,"SeniorCitizen","churn01",1,["tenure","MonthlyCharges","M2M01"]),
      ("telco_partner_drives_tenure",T,"Partner01","tenure",1,["SeniorCitizen","Dependents01"]),
      ("telco_charges_drive_churn",T,"MonthlyCharges","churn01",1,["tenure","SeniorCitizen","M2M01"]),
      ("telco_paperless_drives_churn",T,"Paperless01","churn01",1,["tenure","MonthlyCharges","M2M01"]),
      ("bike_temp_drives_cnt",B,"temp","cnt",1,["hum","windspeed","workingday","yr"]),
      ("bike_hum_drives_cnt",B,"hum","cnt",-1,["temp","windspeed","workingday","yr"]),
      ("bike_windspeed_drives_cnt",B,"windspeed","cnt",-1,["temp","hum","workingday","yr"]),
      ("bike_workingday_drives_cnt",B,"workingday","cnt",1,["temp","hum","windspeed","yr"]),
      ("bike_yr_drives_cnt",B,"yr","cnt",1,["temp","hum","windspeed","workingday"]),
    ]

def complete_sample(df, cols):
    d=df.dropna(subset=cols).reset_index(drop=True); dropped=len(df)-len(d)
    if len(d)>2000: d=d.sample(n=2000,random_state=20261027).reset_index(drop=True)
    return d, dropped

def main():
    frames=build_frames(); specs=edge_specs(*frames)
    builder=json.loads((WORK/"upgrade_probes"/"field_upgrade_results.json").read_text())
    bj={e["id"]:e for e in builder["edges"]}
    published=json.loads((WORK/"probes"/"field_study_results.json").read_text())
    pub={e["id"]:e.get("decision") for e in published["edges"]}
    rows=[]; maxd={"s1":0,"adj":0,"pd":0,"pc":0}
    for eid,fr,xc,yc,d,covs in specs:
        dfr,dropped=complete_sample(fr,[xc,yc]+covs)
        xv=dfr[xc].to_numpy(float); yv=dfr[yc].to_numpy(float)
        C=[dfr[c].to_numpy(float) for c in covs]
        s1=fit(yv,xv,[],d); s2=fit(yv,xv,C,d)
        d1=decide(s1["BF"],s1["Pd"]); d2=decide(s2["BF"],s2["Pd"])
        rat=(d1=="accept" and d2=="accept")
        pc={str(w):p_claim_from_logbf(s2["logBF"],s2["Pd"],w) for w in (0.1,0.5,0.9)}
        b=bj[eid]
        row={"id":eid,"n":int(len(dfr)),"dropped":int(dropped),
             "my_s1_log10BF":s1["log10BF"],"my_s1_dec":d1,"my_adj_log10BF":s2["log10BF"],
             "my_adj_Pd":s2["Pd"],"my_s2_dec":d2,"my_ratified":rat,"my_p_claim_0.5":pc["0.5"],"my_p_claim_all":pc,"my_adj_logBF_nat":s2["logBF"],
             "diff_s1":abs(s1["log10BF"]-b["stage1_log10_BF10"]),
             "diff_adj":abs(s2["log10BF"]-b["adj_log10_BF10"]),
             "diff_adjPd":abs(s2["Pd"]-b["adj_Pd"]),
             "diff_pclaim05":abs(pc["0.5"]-b["p_claim"]["0.5"]),
             "builder_s1_dec":b["stage1_decision"],"builder_s2_dec":b["stage2_decision"],
             "builder_ratified":b["stage2_ratified"],
             "published_decision_for_original7":pub.get(eid)}
        maxd["s1"]=max(maxd["s1"],row["diff_s1"]); maxd["adj"]=max(maxd["adj"],row["diff_adj"])
        maxd["pd"]=max(maxd["pd"],row["diff_adjPd"]); maxd["pc"]=max(maxd["pc"],row["diff_pclaim05"])
        rows.append(row)
        print(eid, row["my_s1_dec"], round(s1["log10BF"],3), "adj", round(s2["log10BF"],3), d2, rat, "diffs", round(row["diff_s1"],6), round(row["diff_adj"],6), flush=True)
    # ledger at 3 w from MY p_claims: full-set expected FDP + prefix keep count
    led={}
    for w in ("0.1","0.5","0.9"):
        rat_rows=[r for r in rows if r["my_ratified"]]
        pcs=[r["my_p_claim_all"][w] for r in rat_rows]
        keep=ledger_keep(pcs,0.05)
        led[w]={"n_ratified":len(rat_rows),"ledger_kept":len(keep),
                "full_set_expected_FDP":float(np.mean([1-p for p in pcs]))}
    counts={"total":len(rows),
            "s1_accept":sum(r["my_s1_dec"]=="accept" for r in rows),
            "s1_reject":sum(r["my_s1_dec"]=="reject" for r in rows),
            "s1_inconclusive":sum(r["my_s1_dec"]=="inconclusive" for r in rows),
            "stage2_ratified":sum(r["my_ratified"] for r in rows),
            "decisions_match_builder_all":all(r["my_s1_dec"]==r["builder_s1_dec"] and r["my_s2_dec"]==r["builder_s2_dec"] and r["my_ratified"]==r["builder_ratified"] for r in rows),
            "original7_published_match":all((r["published_decision_for_original7"] is None) or (r["my_s1_dec"]==r["published_decision_for_original7"]) for r in rows)}
    out={"rows":rows,"max_abs_diffs_vs_builder_json":maxd,"counts":counts,"my_ledger":led}
    Path(__file__).with_name("probe_r3_field_results.json").write_text(json.dumps(out,indent=2))
    print(json.dumps({"max":maxd,"counts":counts},indent=2))

if __name__=="__main__":
    main()
