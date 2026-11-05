#!/usr/bin/env python3
"""R4 field probe: rebuild all 21 field edges with MY corrected core
(1201 grid) using the R3 tester's independent data construction
(probe_r3_field.py build_frames/edge_specs/complete_sample, imported as
construction only -- no builder code, no builder computation).
Computes per-lambda log10BF/Pd/beta/decisions, lambda_min, lambda=1
certification vs cycle-1 adjusted values, and compares every cell to
field2_diagnostic_results.json. Also computes an incoherence diagnostic
for near-singular edges: change in |beta| vs change in log10BF from
lambda=1 to lambda=0.6."""
import json, sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
WORK=HERE.parent
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(WORK/"tester_probes_r3"))
from my_corr_core import fit, decide, lambda_min
from probe_r3_field import build_frames, edge_specs, complete_sample

GRID=(1.0,0.9,0.8,0.7,0.6)
def main():
    O,T,B=build_frames()
    builder=json.loads((WORK/"upgrade2_probes"/"field2_diagnostic_results.json").read_text())
    bj={e["id"]:e for e in builder["edges"]}
    cyc1={c["id"]:c for c in builder["lambda1_certification"]["edges"]}
    rows=[]; maxdiff={"lam1_vs_cycle1":0.0,"vs_builder_anylam_log10bf":0.0,"vs_builder_pd":0.0}
    nmismatch=0
    for eid,frame,xc,yc,d,covs in edge_specs(O,T,B):
        dfr,_=complete_sample(frame,[xc,yc]+covs)
        x=dfr[xc].to_numpy(float); y=dfr[yc].to_numpy(float)
        C=[dfr[c].to_numpy(float) for c in covs]
        lam_map={i:None for i in range(len(covs))}
        per={}; dec={}
        for lam in GRID:
            lm={i:lam for i in range(len(covs))}
            f=fit(y,x,C,d,lam_map=lm,n_grid=1201)
            per[str(lam)]={"log10BF":f.get("log10BF"),"Pd":f.get("Pd"),"beta":f.get("beta_mean"),"evaluable":f.get("evaluable")}
            dec[lam]=decide(f)
            per[str(lam)]["decision"]=dec[lam]
            bcell=bj[eid]["per_lambda"][str(lam)]
            if f.get("log10BF") is not None and bcell.get("log10_BF10") is not None:
                maxdiff["vs_builder_anylam_log10bf"]=max(maxdiff["vs_builder_anylam_log10bf"],abs(f["log10BF"]-bcell["log10_BF10"]))
                maxdiff["vs_builder_pd"]=max(maxdiff["vs_builder_pd"],abs(f["Pd"]-bcell["Pd"]))
                if dec[lam]!=bcell["stage2_decision"]: nmismatch+=1
            elif (f.get("log10BF") is None)!=(bcell.get("log10_BF10") is None):
                nmismatch+=1
        lm_=lambda_min(dec)
        if lm_!=bj[eid]["lambda_min"]: nmismatch+=1
        d1=abs(per["1.0"]["log10BF"]-cyc1[eid]["cycle1_adj_log10_BF10"])
        maxdiff["lam1_vs_cycle1"]=max(maxdiff["lam1_vs_cycle1"],d1)
        # incoherence: beta magnitude change vs BF change 1.0 -> lowest evaluable
        low=[l for l in ("0.6","0.7","0.8") if per[l]["log10BF"] is not None][0]
        inco={"beta_ratio_low_over_1": (abs(per[low]["beta"])/abs(per["1.0"]["beta"]) if per["1.0"]["beta"] else None),
              "dlog10BF_low_minus_1": per[low]["log10BF"]-per["1.0"]["log10BF"], "low_lam":float(low)}
        rows.append({"id":eid,"n":int(len(dfr)),"per_lambda":per,"lambda_min":lm_,
                     "builder_lambda_min":bj[eid]["lambda_min"],"incoherence":inco})
        print(eid, "lmin", lm_, "builder", bj[eid]["lambda_min"],
              " ".join(f"{l}:{per[l]['decision'][:3]}/{per[l]['log10BF'] if per[l]['log10BF'] is None else round(per[l]['log10BF'],2)}" for l in per), flush=True)
    out={"maxdiffs":maxdiff,"decision_or_lmin_mismatches_vs_builder":nmismatch,"rows":rows}
    (HERE/"probe_r4_field_results.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(maxdiff,indent=2),"mismatches",nmismatch)

if __name__=="__main__": main()
