#!/usr/bin/env python3
"""Paper 7 cycle 2 field diagnostic, per upgrade2_preregistration.md
Sections 3 and 6. The 21 cycle-1 field edges (constructions imported
unchanged from cycle-1 run_field_upgrade.py); corrected stage-2 fits
at Lambda = {1.0, 0.9, 0.8, 0.7, 0.6}; lambda=1 certification against
cycle-1 field_upgrade_results.json. DIAGNOSTIC ONLY.
Output: field2_diagnostic_results.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
sys.path.insert(0, str(WORK / "upgrade_probes"))
sys.path.insert(0, str(WORK / "probes"))
sys.path.insert(0, str(HERE))
from corrected_core import fit_corrected, decide_corrected, lambda_min  # noqa: E402
from run_field_upgrade import olist_frame, telco_frame, bike_frame, edge_specs, complete_sample  # noqa: E402

LAM_GRID = (1.0, 0.9, 0.8, 0.7, 0.6)


def main():
    O, T, B = olist_frame(), telco_frame(), bike_frame()
    cycle1 = {e["id"]: e for e in json.loads(
        (WORK / "upgrade_probes" / "field_upgrade_results.json").read_text())["edges"]}
    rows = []
    cert = []
    for eid, frame, xc, yc, d, covs in edge_specs(O, T, B):
        cols = [xc, yc] + covs
        dfr, dropped = complete_sample(frame, cols)
        x = dfr[xc].to_numpy(float); y = dfr[yc].to_numpy(float)
        C = [dfr[c].to_numpy(float) for c in covs]
        per_lam = {}
        dec_by_lam = {}
        for lam in LAM_GRID:
            f = fit_corrected(y, x, C, d, lam)
            per_lam[str(lam)] = {"log10_BF10": f["log10_BF10"], "Pd": f["Pd"],
                                 "beta_mean_std": f["beta_mean_std"],
                                 "evaluable": f["evaluable"],
                                 "stage2_decision": decide_corrected(f)}
            dec_by_lam[lam] = per_lam[str(lam)]["stage2_decision"]
        lm = lambda_min(dec_by_lam)
        c1 = cycle1[eid]
        disc = abs(per_lam["1.0"]["log10_BF10"] - c1["adj_log10_BF10"]) \
            if per_lam["1.0"]["log10_BF10"] is not None else None
        cert.append({"id": eid, "cycle2_lam1_log10_BF10": per_lam["1.0"]["log10_BF10"],
                     "cycle1_adj_log10_BF10": c1["adj_log10_BF10"],
                     "abs_discrepancy": disc})
        row = {"id": eid, "direction": d, "covariates": covs, "n": int(len(dfr)),
               "stage1_decision_cycle1": c1["stage1_decision"],
               "stage2_ratified_cycle1": c1["stage2_ratified"],
               "per_lambda": per_lam, "lambda_min": lm}
        rows.append(row)
        print(f"{eid:34s} " + " ".join(
            f"L{lam}={per_lam[str(lam)]['log10_BF10']:+.2f}/{per_lam[str(lam)]['stage2_decision'][:3]}"
            if per_lam[str(lam)]["log10_BF10"] is not None else f"L{lam}=NE"
            for lam in LAM_GRID) + f" lam_min={lm}", flush=True)
    max_disc = max(c["abs_discrepancy"] for c in cert if c["abs_discrepancy"] is not None)
    out = {"preregistration": "upgrade2_preregistration.md Sections 3 and 6",
           "diagnostic_only": True,
           "lambda1_certification": {"edges": cert, "max_abs_discrepancy": float(max_disc),
                                     "criterion": "<= 0.01 log10 on each edge (21 edges computed)"},
           "edges": rows}
    (HERE / "field2_diagnostic_results.json").write_text(json.dumps(out, indent=2))
    print("lambda=1 certification max |disc| =", max_disc, "on", len(cert), "edges")


if __name__ == "__main__":
    main()
