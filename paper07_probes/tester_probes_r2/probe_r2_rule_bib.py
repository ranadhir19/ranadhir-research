#!/usr/bin/env python3
"""R2 tester probe 2: rule-of-record grid, shipped-JSON recomputation,
bibliography audit, manuscript table-vs-JSON cell checks. Tester-written."""
import json, re, subprocess, sys
from pathlib import Path
BASE = Path(__file__).resolve().parents[1]
PROBES = BASE/"probes"
TEX = Path.home()/"workspace/goals/10-journal-papers-submitted-to-good-journals/files/Paper_07_BayesianKPI_DataKnowledgeEngineering.tex"
sys.path.insert(0, str(PROBES))
from rule_of_record import decide, _old_code_decide

out={}
# --- run builder unit tests as a subprocess
r = subprocess.run([sys.executable, str(PROBES/"rule_of_record.py")], capture_output=True, text=True)
out["rule_tests_stdout"]=r.stdout.strip(); out["rule_tests_rc"]=r.returncode

# --- tester's own 63-cell grid (same axes as rule_of_record __main__)
bfs=[0.1,0.34,1.0,2.0,3.0,5.0,9.9,10.0,100.0]; pds=[0.0,0.5,0.89,0.90,0.94,0.95,0.99]
div=[(bf,pd,decide(bf,pd),_old_code_decide(bf,pd)) for bf in bfs for pd in pds if decide(bf,pd)!=_old_code_decide(bf,pd)]
out["grid"]=dict(n_cells=len(bfs)*len(pds), n_divergent=len(div), divergent=[list(x) for x in div],
                 all_divergent_are_new_accepts=all(x[2]=="accept" for x in div))
print(f"grid: {len(bfs)*len(pds)} cells, divergences={len(div)}, all new-accepts: {out['grid']['all_divergent_are_new_accepts']}")
for x in div: print("  div", x)

# --- alternative grid incl. Pd=1.0 and BF=1e6 (round-1-style sensitivity)
bfs2=[0.1,1/3,0.34,1.0,3.0,5.0,10.0,1e6]; pds2=[0.0,0.05,0.5,0.90,0.95,1.0]
div2=[(bf,pd) for bf in bfs2 for pd in pds2 if decide(bf,pd)!=_old_code_decide(bf,pd)]
out["grid_alt"]=dict(n_cells=len(bfs2)*len(pds2), n_divergent=len(div2))
print(f"alt grid: {len(bfs2)*len(pds2)} cells, divergences={len(div2)}")

# --- shipped replication JSON: recompute decisions under rule of record
ship=json.loads((PROBES/"kpi_governance_results.json").read_text())
edges = ship["edges"] if isinstance(ship, dict) and "edges" in ship else ship
recomp=[]; 
for e in edges:
    bf=e.get("bf10"); pd_=e.get("p_dir")
    if bf is None or pd_ is None: recomp.append((e.get("slug"),"MISSING FIELDS",list(e.keys()))); continue
    recomp.append((e.get("slug"), decide(float(bf), float(pd_)), e.get("decision")))
mismatch=[x for x in recomp if x[1]!=x[2]]
out["shipped_recompute"]=dict(n=len(recomp), mismatches=mismatch, rows=recomp)
print(f"shipped JSON: {len(recomp)} edges recomputed, mismatches={mismatch}")

# --- strengthening results audit (logic-level)
sc=json.loads((PROBES/"strengthening_checks_results.json").read_text())
out["strengthening"]=dict(stable=sc["multiseed_rule_check"]["stable_across_seeds"],
    rules_agree=sc["multiseed_rule_check"]["rules_agree_per_seed"],
    canonical_counts={k:list(sc["multiseed_rule_check"]["canonical_vector"].values()).count(k) for k in ["accept","reject","inconclusive"]},
    confounder=[(c["seed"], round(c["beta_mean"],3), c["decision"]) for c in sc["confounder_control"]])
print("strengthening:", out["strengthening"])

# --- bibliography audit on the .tex
tex=TEX.read_text()
cited=set()
for m in re.finditer(r"\\cite\{([^}]*)\}", tex):
    for k in m.group(1).split(","): cited.add(k.strip())
bib=set(re.findall(r"\\bibitem\{([^}]*)\}", tex))
out["bib"]=dict(n_cited=len(cited), n_bib=len(bib), cited_not_in_bib=sorted(cited-bib), bib_not_cited=sorted(bib-cited))
print("bib:", out["bib"])

# --- field table cell check vs builder JSON (rounding to table precision)
fj={e["id"]:e for e in json.loads((PROBES/"field_study_results.json").read_text())["edges"]}
def rnd(v,nd=3): return round(float(v)+0.0, nd)
checks=[]
table = {  # transcribed from manuscript tab:field
 "olist_delay_drives_review":      (-0.027, (-0.032,-0.023), 1.000, 19.87, 1.003, "accept"),
 "olist_freight_drives_review":    (-0.007, (-0.010,-0.004), 1.000, 1.23, 1.001, "accept"),
 "olist_installments_drive_value": (24.01, (20.80,27.43), 1.000, 35.74, 1.002, "accept"),
 "olist_zip_null_control":         (0.000, (0.000,0.000), 0.915, -4.28, 1.003, "reject"),
 "telco_tenure_drives_charges":    (0.334, (0.284,0.386), 1.000, 29.36, 1.003, "accept"),
 "telco_tenure_drives_churn":      (-0.006, (-0.007,-0.005), 1.000, 48.85, 1.001, "accept"),
 "telco_charges_drive_total":      (46.31, (43.95,48.66), 1.000, 219.50, 1.000, "accept"),
}
for k,(b,h,pd_,lbf,rh,dec) in table.items():
    j=fj[k]
    row=dict(edge=k,
      beta_ok = rnd(j["beta_mean"],3 if abs(b)<10 else 2)==b or abs(rnd(j["beta_mean"],2)-b)<1e-9,
      beta_json=j["beta_mean"], beta_table=b,
      hdi_lo_json=j["hdi94"][0], hdi_lo_table=h[0], hdi_lo_round3=rnd(j["hdi94"][0]),
      hdi_hi_json=j["hdi94"][1], hdi_hi_table=h[1], hdi_hi_round3=rnd(j["hdi94"][1]),
      pd_json=j["Pd"], pd_table=pd_,
      logbf_json=j["bridge_log10_BF10"], logbf_table=lbf, logbf_round2=rnd(j["bridge_log10_BF10"],2),
      rhat_json=j["rhat_beta"], rhat_table=rh,
      dec_json=j["decision"], dec_table=dec)
    checks.append(row)
    print(row)
out["field_table_checks"]=checks
(Path(__file__).parent/"probe_r2_rule_bib_results.json").write_text(json.dumps(out,indent=2,default=str))
print("wrote probe_r2_rule_bib_results.json")
