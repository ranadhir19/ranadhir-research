#!/usr/bin/env python3
"""R3 table audit v2: manuscript Tables field2a/field2b vs field JSON, cell by
cell at printed precision, line-based parsing. Also simcal table vs sim JSON
and the ledger expected-FDP numbers in the text."""
import json, re
from pathlib import Path
import numpy as np
W=Path(__file__).parent.parent; G=W.parent.parent
tex=(G/"files"/"Paper_07_BayesianKPI_DataKnowledgeEngineering.tex").read_text()
fj=json.loads((W/"upgrade_probes"/"field_upgrade_results.json").read_text())
by={e["id"]:e for e in fj["edges"]}
label2id={
 "olist_delay_review":"olist_delay_drives_review","olist_freight_review":"olist_freight_drives_review",
 "olist_installments_value":"olist_installments_drive_value","olist_zip_control":"olist_zip_null_control",
 "olist_nitems_freight":"olist_nitems_drive_freight","olist_freight_delay":"olist_freight_drives_delay",
 "olist_review_freight":"olist_review_drives_freight","olist_statecode_review":"olist_statecode_drives_review",
 "olist_nitems_review":"olist_nitems_drive_review","telco_tenure_charges":"telco_tenure_drives_charges",
 "telco_tenure_churn":"telco_tenure_drives_churn","telco_charges_total":"telco_charges_drive_total",
 "telco_senior_churn":"telco_senior_drives_churn","telco_partner_tenure":"telco_partner_drives_tenure",
 "telco_charges_churn":"telco_charges_drive_churn","telco_paperless_churn":"telco_paperless_drives_churn",
 "bike_temp_cnt":"bike_temp_drives_cnt","bike_hum_cnt":"bike_hum_drives_cnt",
 "bike_windspeed_cnt":"bike_windspeed_drives_cnt","bike_workingday_cnt":"bike_workingday_drives_cnt",
 "bike_yr_cnt":"bike_yr_drives_cnt"}
out={}
lines=[l.strip() for l in tex.splitlines() if l.strip().startswith("\\texttt{") and "&" in l]
bad=[]; n=0
for l in lines:
    m=re.match(r"\\texttt\{([a-z_0-9\\]+)\}", l)
    if not m: continue
    lab=m.group(1).replace("\\","")
    if lab not in label2id: continue
    n+=1
    body=l.rstrip("\\")
    cells=[c.strip().strip("$") for c in body.split("&")]
    got=[cells[1], cells[2].strip("$"), cells[3], cells[4].replace("$^{\\dagger}$","").strip(), cells[5], cells[6].replace("\\\\","").strip()]
    eid=label2id[lab]; b=by[eid]
    exp=[b["stage1_decision"], f"{b['adj_log10_BF10']:.2f}", f"{b['adj_Pd']:.3f}",
         "ratified" if b["stage2_ratified"] else ("not ratified" if eid=="bike_hum_drives_cnt" else b["stage2_decision"]),
         f"{b['p_claim']['0.5']:.3f}", "ratified" if b.get("ledger_ratified",{}).get("0.5") else "--"]
    if got!=exp: bad.append({"edge":eid,"got":got,"expected":exp})
out["field_table_rows_parsed"]=n; out["field_table_mismatches"]=bad
# simcal table: three rows in tex
sj=json.loads((W/"upgrade_probes"/"simulation_results.json").read_text())["summary"]
def fmt(proc):
    s=sj[proc]
    return (f"{s['ratified']['mean']:.1f} ({s['ratified']['median']:.0f}; {s['ratified']['min']:.0f}--{s['ratified']['max']:.0f})",
            f"{s['false_ratified']['mean']:.1f}", f"{s['true_ratified']['mean']:.1f}",
            f"{s['fdp']['mean']:.3f} ({s['fdp']['median']:.3f}; {s['fdp']['min']:.3f}--{s['fdp']['max']:.3f})")
out["simcal_json_formatted"]={p:fmt(p) for p in "abc"}
for needle in ("26.1 (26; 22--28)","14.3 (14.5; 10--16)","0.446 (0.444; 0.364--0.522)"):
    out[f"tex_contains_{needle}"]= needle in tex
rat=[e for e in fj["edges"] if e["stage2_ratified"]]
out["ledger_expFDP_full13"]={w: float(np.mean([1-e["p_claim"][w] for e in rat])) for w in ("0.1","0.5","0.9")}
out["freight_pclaim"]={w: [e["p_claim"][w] for e in rat if e["id"]=="olist_freight_drives_review"][0] for w in ("0.1","0.5","0.9")}
out["counts"]=fj["counts"]; out["ledger_counts"]={w:fj["ledger_summary"][w]["ledger_ratified_count"] for w in ("0.1","0.5","0.9")}
Path(__file__).with_name("probe_r3_tables_results.json").write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
