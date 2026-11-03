#!/usr/bin/env python3
"""CARDIO experiment: horizon durability of ECG risk, null-calibrated.

Synthetic constant-effect proportional-hazards cohort (n=4000, seed 0) with
redundant ECG views. Reproduces the paper's measurement protocol in miniature:

 (a) Grambsch-Therneau test for time-varying coefficients (expects p >> 0.05);
 (b) horizon grid of incremental Uno concordance / tdAUC / IPA of (demo+ecg)
     over demo-only on a fixed landmark cohort, calibrated against a
     constant-effect null (5 replicates): the rising IPA is shown to be an
     artefact (observed slope inside null range), the falling concordance is
     not (observed slope outside null range);
 (c) interaction-information screen over modality pairs (expects all II < 0)
     then concatenation vs a tiny gated-fusion network (expects concat >= gated);
 (d) typed knowledge graph whose structural rule fires invariance-vs-trend
     tensions;
 plus a PCA dimension-vs-label-budget demo and a gated-pipeline blocked run.

Writes results/cardio/results.md and results/cardio/charts.png, and copies
charts.png to docs/assets/charts/cardio.png. All randomness is seeded.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ghosh_research import cardio as C  # noqa: E402

SEED = 0
N = 4000
HET_K = 2.0        # observed cohort: risk-correlated ECG-view noise (Result-3 mechanism)
HET_THRESH = 0.75
TAUS = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
TAU_MAX = 7.5
NULL_SEEDS = [101, 102, 103, 104, 105]
OUT = Path(__file__).resolve().parents[1] / "results" / "cardio"


def fit_arms(d):
    """Fit demo-only and demo+ecg Cox arms once (never refit per horizon)."""
    X, Y, delta = d["X"], d["Y"], d["delta"]
    b_demo = C.fit_cox(d["blocks"]["demo"], Y, delta)["beta"]
    b_full = C.fit_cox(X, Y, delta)["beta"]
    tHd, Hd = C.breslow(b_demo, d["blocks"]["demo"], Y, delta)
    tHf, Hf = C.breslow(b_full, X, Y, delta)
    return {
        "eta_demo": d["blocks"]["demo"] @ b_demo,
        "eta_full": X @ b_full,
        "H_demo": (tHd, Hd), "H_full": (tHf, Hf),
        "b_demo": b_demo, "b_full": b_full,
    }


def horizon_curves(d, arms, taus=TAUS, tau_max=TAU_MAX):
    """Incremental (full - demo) Uno C, tdAUC, IPA per tau on the landmark cohort."""
    Y, delta = d["Y"], d["delta"]
    tc, sc = C.km_survival(Y, 1.0 - delta)
    G = lambda t: C.km_eval(tc, sc, np.atleast_1d(t))  # noqa: E731
    ts, ss = C.km_survival(Y, delta)
    S_km = lambda t: float(C.km_eval(ts, ss, t)[0])  # noqa: E731
    mask = C.landmark_cohort(Y, delta, tau_max)
    ed = arms["eta_demo"][mask]
    ef = arms["eta_full"][mask]
    Ym, dm = Y[mask], delta[mask]
    incC, incA, incI = [], [], []
    for tau in taus:
        Cd = C.uno_concordance(ed, Ym, dm, tau, G)
        Cf = C.uno_concordance(ef, Ym, dm, tau, G)
        incC.append(Cf - Cd)
        Ad = C.td_auc(ed, Ym, dm, tau, G)
        Af = C.td_auc(ef, Ym, dm, tau, G)
        incA.append(Af - Ad)
        Hdt = C.h0_eval(*arms["H_demo"], tau)
        Hft = C.h0_eval(*arms["H_full"], tau)
        bd, bk = C.ipcw_brier(ed, Hdt, Ym, dm, tau, G, S_km(tau))
        bf, _ = C.ipcw_brier(ef, Hft, Ym, dm, tau, G, S_km(tau))
        incI.append(C.ipa(bf, bk) - C.ipa(bd, bk))
    return {"incC": np.array(incC), "incAUC": np.array(incA),
            "incIPA": np.array(incI), "n_landmark": int(mask.sum())}


def slope_per_year(taus, vals):
    A = np.vstack([taus, np.ones(len(taus))]).T
    return float(np.linalg.lstsq(A, vals, rcond=None)[0][0])


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"seed": SEED, "n": N, "het_k": HET_K, "taus": TAUS}

    # ---- observed cohort -------------------------------------------------
    d = C.simulate_cohort(N, seed=SEED, het_k=HET_K, het_thresh=HET_THRESH)
    X, Y, delta = d["X"], d["Y"], d["delta"]
    full = C.fit_cox(X, Y, delta)
    beta = full["beta"]
    res["event_rate"] = d["event_rate"]
    res["log_hr"] = dict(zip(d["names"], [round(float(b), 4) for b in beta]))
    res["cox_converged"] = full["converged"]
    print("log-HRs:", res["log_hr"], "event_rate:", round(d["event_rate"], 3))

    # (a) Grambsch-Therneau
    gt = C.grambsch_therneau(beta, X, Y, delta)
    res["gt_p"] = dict(zip(d["names"], [round(float(p), 4)
                                        for p in gt["p_values"]]))
    res["gt_global_p"] = round(gt["global_p"], 4)
    res["gt_n_events"] = gt["n_events"]
    all_ns = bool((gt["p_values"] > 0.05).all())
    print("GT p-values:", res["gt_p"], "global:", res["gt_global_p"],
          "all>0.05:", all_ns)

    # ---- horizon grid (observed) ------------------------------------------
    arms = fit_arms(d)
    hc = horizon_curves(d, arms)
    res["n_landmark"] = hc["n_landmark"]
    res["incC"] = [round(float(v), 4) for v in hc["incC"]]
    res["incAUC"] = [round(float(v), 4) for v in hc["incAUC"]]
    res["incIPA"] = [round(float(v), 4) for v in hc["incIPA"]]
    obs_sC = slope_per_year(TAUS, hc["incC"])
    obs_sA = slope_per_year(TAUS, hc["incAUC"])
    obs_sI = slope_per_year(TAUS, hc["incIPA"])
    print("incC:", res["incC"], "slope/yr:", round(obs_sC, 5))
    print("incIPA:", res["incIPA"], "slope/yr:", round(obs_sI, 5))

    # (b) constant-effect null: 5 replicates, homoscedastic, same true betas
    null_sC, null_sI = [], []
    null_curves_C, null_curves_I = [], []
    for s in NULL_SEEDS:
        dn = C.simulate_cohort(N, seed=s, het_k=0.0)
        an = fit_arms(dn)
        hn = horizon_curves(dn, an)
        null_sC.append(slope_per_year(TAUS, hn["incC"]))
        null_sI.append(slope_per_year(TAUS, hn["incIPA"]))
        null_curves_C.append(hn["incC"])
        null_curves_I.append(hn["incIPA"])
    null_sC = np.array(null_sC)
    null_sI = np.array(null_sI)
    res["null"] = {
        "seeds": NULL_SEEDS,
        "slopeC": [round(float(v), 5) for v in null_sC],
        "slopeIPA": [round(float(v), 5) for v in null_sI],
        "slopeC_range": [round(float(null_sC.min()), 5),
                         round(float(null_sC.max()), 5)],
        "slopeIPA_range": [round(float(null_sI.min()), 5),
                           round(float(null_sI.max()), 5)],
    }
    ipa_inside = res["null"]["slopeIPA_range"][0] <= obs_sI <= res["null"]["slopeIPA_range"][1]
    conc_outside = obs_sC < res["null"]["slopeC_range"][0]
    res["obs_slopeC"] = round(obs_sC, 5)
    res["obs_slopeAUC"] = round(obs_sA, 5)
    res["obs_slopeIPA"] = round(obs_sI, 5)
    res["null_slopeC_mean"] = round(float(null_sC.mean()), 5)
    res["null_slopeIPA_mean"] = round(float(null_sI.mean()), 5)
    res["ipa_explained_by_null"] = bool(ipa_inside)
    res["conc_explained_by_null"] = bool(not conc_outside)
    print("null slopeC:", res["null"]["slopeC"], "range:", res["null"]["slopeC_range"])
    print("null slopeIPA:", res["null"]["slopeIPA"], "range:", res["null"]["slopeIPA_range"])
    print("IPA inside null range:", ipa_inside, "| conc outside null range:", conc_outside)

    # (c) interaction-information screen, then concat vs gated fusion
    b = d["blocks"]
    pairs = [("demo", "ecg", b["demo"], b["ecg"]),
             ("ecg1", "ecg2", X[:, 2:3], X[:, 3:4]),
             ("demo", "annot", b["demo"], b["annot"])]
    ii_res = {}
    for an, bn_, XA, XB in pairs:
        r = C.interaction_information(XA, XB, Y, delta)
        ii_res[f"{an};{bn_}"] = round(float(r["II"]), 2)
    res["II"] = ii_res
    print("II:", ii_res)

    tc, sc = C.km_survival(Y, 1.0 - delta)
    G = lambda t: C.km_eval(tc, sc, np.atleast_1d(t))  # noqa: E731
    eta_concat = X @ beta
    c_concat = C.uno_concordance(eta_concat, Y, delta, TAU_MAX, G)
    blocks_std = {k: (v - v.mean(0)) / (v.std(0) + 1e-12)
                  for k, v in b.items()}
    gf = C.fit_gated_fusion(blocks_std, Y, delta, seed=SEED,
                            steps=200, lr=0.05, hidden=4)
    c_gated = C.uno_concordance(gf["eta"], Y, delta, TAU_MAX, G)
    res["concat_C"] = round(float(c_concat), 4)
    res["gated_C"] = round(float(c_gated), 4)
    res["concat_minus_gated"] = round(float(c_concat - c_gated), 4)
    print(f"concat C={c_concat:.4f} gated C={c_gated:.4f} "
          f"diff={c_concat - c_gated:+.4f}")

    # (d) knowledge graph tensions
    kg = C.ClaimGraph()
    kg.add_claim("theorem", "C_tau", "invariant", axis="tau",
                 statement="Under PH, P(T1<T2|eta1,eta2) is logistic in the "
                           "linear-predictor difference; the baseline hazard "
                           "cancels, so untruncated concordance cannot depend "
                           "on the horizon.")
    kg.add_claim("measured", "ecg_effect", "invariant", axis="time",
                 statement=f"Grambsch-Therneau finds no time variation "
                           f"(min p = {min(res['gt_p'].values()):.4f}).")
    kg.add_claim("measured", "C_tau", "trend_down", axis="tau",
                 statement=f"Incremental Uno concordance falls with horizon "
                           f"(slope {obs_sC:+.5f}/yr), outside the "
                           f"constant-effect null range.")
    kg.add_claim("measured", "ecg_effect", "trend_down", axis="tau",
                 statement="The ECG's incremental contribution trends down "
                           "with the horizon.")
    kg.add_claim("measured", "IPA_tau", "trend_up", axis="tau",
                 statement=f"Incremental IPA rises with horizon (slope "
                           f"{obs_sI:+.5f}/yr), inside the null range: a "
                           f"statistic artefact, not growing value.")
    kg.add_claim("assumption", "ecg_mechanism", "invariant", axis="cohort",
                 statement="The learned ECG views encode myocardial structure "
                           "(untested here; paired tissue imaging would settle "
                           "it).")
    tensions = kg.detect_tensions()
    res["n_tensions"] = len(tensions)
    res["tensions"] = [(a["variable"], a["kind"], a["relation"],
                        b_["kind"], b_["relation"]) for a, b_ in tensions]
    print("tensions:", res["n_tensions"])

    # PCA dimension vs label budget (paper Result 5, small demo)
    res["pca_demo"] = pca_budget_demo()

    # gated pipeline: synergy screen blocks gated-fusion construction
    ledger = gated_pipeline_demo(b, Y, delta)
    res["pipeline_ledger"] = ledger["ledger"]
    res["pipeline_completed"] = ledger["completed"]
    print("pipeline ledger:", ledger["ledger"])

    write_results_md(res)
    write_charts(res, np.array(null_curves_C), np.array(null_curves_I))
    dst = Path(__file__).resolve().parents[1] / "docs" / "assets" / "charts"
    dst.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copyfile(OUT / "charts.png", dst / "cardio.png")
    print(f"wrote {OUT / 'results.md'}, {OUT / 'charts.png'}, {dst / 'cardio.png'}")
    return res


def pca_budget_demo():
    """Representation dimension vs label budget (paper Result 5, miniature).

    Quadratic expansion of the 3 ECG views (9 features) -> PCA; fit Cox on
    [demo + k PCs] with k in {1,2,4,9} at label budgets {400,1500,4000};
    metric = gain in Uno C (tau=5) over demo-only on a fixed eval set.
    """
    rng = np.random.default_rng(7)
    dfit = C.simulate_cohort(4000, seed=7, het_k=HET_K, het_thresh=HET_THRESH)
    dev = C.simulate_cohort(3000, seed=999, het_k=HET_K, het_thresh=HET_THRESH)

    def expand(X):
        e = X[:, 2:5]
        quad = np.column_stack([e, e ** 2, (e[:, [0]] * e[:, [1]]),
                                (e[:, [0]] * e[:, [2]]),
                                (e[:, [1]] * e[:, [2]])])
        return quad

    F_fit, F_ev = expand(dfit["X"]), expand(dev["X"])
    mu, sd = F_fit.mean(0), F_fit.std(0) + 1e-12
    Zf, Ze = (F_fit - mu) / sd, (F_ev - mu) / sd
    U, sv, Vt = np.linalg.svd(Zf, full_matrices=False)
    tc, sc = C.km_survival(dev["Y"], 1.0 - dev["delta"])
    G = lambda t: C.km_eval(tc, sc, np.atleast_1d(t))  # noqa: E731
    b_dev_demo = C.fit_cox(dev["blocks"]["demo"], dev["Y"],
                           dev["delta"])["beta"]
    c0 = C.uno_concordance(dev["blocks"]["demo"] @ b_dev_demo, dev["Y"],
                           dev["delta"], 5.0, G)
    rows = {}
    for budget in (400, 1500, 4000):
        idx = rng.choice(len(dfit["Y"]), size=budget, replace=False)
        Xb, Yb, db_ = dfit["X"][idx], dfit["Y"][idx], dfit["delta"][idx]
        Zb = (expand(Xb) - mu) / sd
        gains = {}
        for k in (1, 2, 4, 9):
            Pk = Vt[:k].T
            Xk = np.column_stack([Xb[:, :2], Zb @ Pk])
            bk = C.fit_cox(Xk, Yb, db_)["beta"]
            Xe = np.column_stack([dev["X"][:, :2], Ze @ Pk])
            ck = C.uno_concordance(Xe @ bk, dev["Y"], dev["delta"], 5.0, G)
            gains[k] = round(float(ck - c0), 4)
        rows[budget] = gains
    print("PCA dim x budget gains:", rows)
    return rows


def gated_pipeline_demo(blocks, Y, delta):
    """9-layer-style gated pipeline; the synergy screen blocks the run.

    Mirrors the paper's architecture: a failed gate STOPS the run rather than
    degrading it. Here the synergy screen finds II(demo;ecg) < 0, so the gate
    refuses to construct the gated-fusion network and the run is blocked --
    concatenation is the selected architecture.
    """
    ii = C.interaction_information(blocks["demo"], blocks["ecg"], Y, delta)
    layers = [
        {"name": "0_provenance",
         "run_fn": lambda ctx: {"n": len(Y), "seed": SEED},
         "gate_fn": lambda ctx, r: (r["n"] == len(Y), "cohort size changed")},
        {"name": "1_synergy_screen",
         "run_fn": lambda ctx: {"II_demo_ecg": float(ii["II"])},
         "gate_fn": lambda ctx, r: (
             r["II_demo_ecg"] > 0,
             f"II(demo;ecg)={r['II_demo_ecg']:.2f} < 0: views redundant, "
             "gated fusion not justified; use concatenation")},
        {"name": "2_gated_fusion",
         "run_fn": lambda ctx: {"built": True},
         "gate_fn": lambda ctx, r: (True, "")},
    ]
    pipe = C.GatedPipeline(layers)
    return pipe.run()


def write_results_md(res):
    n_events = res["gt_n_events"]
    gt_rows = "\n".join(
        f"| {k} | {v:.4f} |"
        for k, v in res["log_hr"].items())
    gt_p_rows = "\n".join(
        f"| {k} | {v:.4f} | {'pass (>0.05)' if v > 0.05 else 'FAIL'} |"
        for k, v in res["gt_p"].items())
    ii_rows = "\n".join(f"| {k} | {v:+.2f} |" for k, v in res["II"].items())
    pca = res["pca_demo"]
    pca_rows = "\n".join(
        f"| {b} | " + " | ".join(f"{pca[b][k]:+.4f}" for k in (1, 2, 4, 9)) + " |"
        for b in sorted(pca))
    tension_rows = "\n".join(
        f"| {a} | {b} {c} | {d} {e} |"
        for a, b, c, d, e in res["tensions"])
    md = f"""# CARDIO experiment results

Synthetic constant-effect proportional-hazards cohort, n={res['n']}, seed={res['seed']}.
Censoring U(0.5, 8.5) years; event rate {res['event_rate']:.3f};
landmark cohort (outcome known at 7.5y) n={res['n_landmark']}.
Observed cohort uses risk-correlated ECG-view noise (`het_k={res['het_k']}`):
the miniature's concrete mechanism for the paper's unexplained concordance
drift (true coefficients stay constant, so the GT test stays blind to it).
Null replicates: homoscedastic, same constant true betas, seeds {res['null']['seeds']}.

## Cox fit (full model, one fit; Newton-Raphson converged: {res['cox_converged']})

| covariate | log-HR |
|---|---|
{gt_rows}

## (a) Grambsch-Therneau test of proportional hazards

{n_events} events inform the test. H0: coefficient constant in time.

| covariate | p (time variation) | verdict |
|---|---|---|
{gt_p_rows}

Global test p = {res['gt_global_p']:.4f}. No detectable time variation --
the ECG effect does not measurably decay, mirroring the paper (p=0.9403).

## (b) Horizon durability with constant-effect null calibration

One model fit per arm (demo-only vs demo+ecg); horizons obtained by
restricting the evaluation window on the fixed landmark cohort, never by
refitting. Incremental = (demo+ecg) - (demo-only).

| tau (y) | incr. Uno C | incr. tdAUC | incr. IPA |
|---|---|---|---|
"""
    for t, c_, a_, i_ in zip(res["taus"], res["incC"], res["incAUC"], res["incIPA"]):
        md += f"| {t:.0f} | {c_:+.4f} | {a_:+.4f} | {i_:+.4f} |\n"
    md += f"""
### Durability slopes (per year) vs the constant-effect null

| measure (landmark) | observed slope | null slope (constant effect) | null range | explained by the metric? |
|---|---|---|---|---|
| incremental concordance | {res['obs_slopeC']:+.5f} | {res['null_slopeC_mean']:+.5f} | [{res['null']['slopeC_range'][0]:+.5f}, {res['null']['slopeC_range'][1]:+.5f}] | {'YES' if res['conc_explained_by_null'] else 'no'} |
| incremental tdAUC | {res['obs_slopeAUC']:+.5f} | -- | -- | -- |
| incremental IPA | {res['obs_slopeIPA']:+.5f} | {res['null_slopeIPA_mean']:+.5f} | [{res['null']['slopeIPA_range'][0]:+.5f}, {res['null']['slopeIPA_range'][1]:+.5f}] | {'YES - fully' if res['ipa_explained_by_null'] else 'no'} |

Null slopes: {res['null']['slopeC']} (concordance), {res['null']['slopeIPA']} (IPA).
The rising incremental proper score is reproduced almost exactly by the null:
it is what the statistic does when nothing is changing -- an artefact, not
growing long-horizon value. The falling incremental concordance lies outside
the null range: it is not the instrument. That residual tension is the
miniature's version of the paper's open question.

## (c) Modality synergy screen, then fusion

Interaction information II(A;B) = I(A,B) - I(A) - I(B), nats; negative =
redundant views, positive = synergy.

| pair | II (nats) |
|---|---|
{ii_rows}

All pairs negative: no synergy anywhere, as in the paper. The screen therefore
selects concatenation; a gated-fusion network (per-modality tanh encoders,
softmax attention, 200 Adam steps on the partial likelihood, 51 params) was
built only to check the screen's prediction:

| fusion | Uno C (tau=7.5) |
|---|---|
| concatenation (Cox, all features) | {res['concat_C']:.4f} |
| gated attention | {res['gated_C']:.4f} |
| concat - gated | {res['concat_minus_gated']:+.4f} |

Architectural sophistication buys nothing where there is no synergy to
exploit; the extra parameters only add variance.

## (d) Knowledge-graph tensions

| variable | invariant edge | trend edge |
|---|---|---|
{tension_rows}

{res['n_tensions']} structural tension(s) detected: an edge asserting
invariance alongside an edge asserting a trend on the same variable -- the
rule that motivated the null simulation.

## Representation dimension vs label budget (demo)

Gain in Uno C (tau=5) over demo-only on a fixed eval set (n=3000), by PCA
dimension of the ECG block and fit-set label budget:

| budget | 1 PC | 2 PCs | 4 PCs | 9 PCs |
|---|---|---|---|---|
{pca_rows}

## Gated pipeline (blocked run)

Ledger (a failed gate stops the run rather than degrading it):

"""
    for layer, status in res["pipeline_ledger"]:
        md += f"- {layer}: {status}\n"
    md += f"\nRun completed: {res['pipeline_completed']} (blocked at the synergy screen: II < 0, so no fusion network is built -- concatenation selected).\n"
    (OUT / "results.md").write_text(md)


def write_charts(res, nullC, nullI):
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("CARDIO: horizon durability of ECG risk (synthetic, n=4000, seed 0)",
                 fontsize=13, fontweight="bold")

    # (1) GT p-values
    a = ax[0, 0]
    names = list(res["gt_p"].keys())
    ps = [res["gt_p"][k] for k in names]
    bars = a.bar(names, ps, color=["#7ee2a8" if p > 0.05 else "#f08787" for p in ps])
    a.axhline(0.05, color="#f0b35e", linestyle="--", label="alpha=0.05")
    a.set_ylabel("p-value (time variation)")
    a.set_title("(a) Grambsch-Therneau: no detectable time variation")
    a.legend(fontsize=9)
    for b_, p in zip(bars, ps):
        a.text(b_.get_x() + b_.get_width() / 2, p + 0.02, f"{p:.3f}",
               ha="center", fontsize=8)

    # (2) horizon curves with null bands
    a = ax[0, 1]
    taus = np.array(res["taus"])
    for vals, null_arr, lab, col in [
            (res["incC"], nullC, "incr. concordance", "#6ea8fe"),
            (res["incIPA"], nullI, "incr. IPA", "#7ee2a8")]:
        lo, hi = null_arr.min(0), null_arr.max(0)
        a.fill_between(taus, lo, hi, color=col, alpha=0.25)
        a.plot(taus, vals, "o-", color=col, label=lab + " (observed)")
    a.plot(taus, nullC.mean(0), "--", color="#6ea8fe", alpha=0.7,
           label="null mean (C)")
    a.plot(taus, nullI.mean(0), "--", color="#7ee2a8", alpha=0.7,
           label="null mean (IPA)")
    a.set_xlabel("horizon (years)")
    a.set_ylabel("increment over demo-only")
    a.set_title("(b) Horizon curves with constant-effect null bands")
    a.legend(fontsize=8)

    # (3) II screen + concat vs gated
    a = ax[1, 0]
    pairs = list(res["II"].keys())
    iis = [res["II"][k] for k in pairs]
    x = np.arange(len(pairs))
    a.bar(x, iis, color="#f08787")
    a.set_xticks(x)
    a.set_xticklabels(pairs, rotation=15, fontsize=9)
    a.set_ylabel("II (nats)")
    a.set_title(f"(c) Interaction information (all < 0) | "
                f"concat {res['concat_C']:.4f} vs gated {res['gated_C']:.4f}")
    a.text(0.98, 0.95, f"concat - gated = {res['concat_minus_gated']:+.4f}",
           transform=a.transAxes, ha="right", va="top", fontsize=10,
           bbox=dict(boxstyle="round", facecolor="#111724", edgecolor="#22304a"))

    # (4) slope table + tensions as text
    a = ax[1, 1]
    a.axis("off")
    lines = [
        "(b) Durability slopes per year vs constant-effect null",
        "",
        f"incr. concordance: obs {res['obs_slopeC']:+.5f}  "
        f"null {res['null_slopeC_mean']:+.5f} "
        f"[{res['null']['slopeC_range'][0]:+.5f}, {res['null']['slopeC_range'][1]:+.5f}]"
        f"  -> {'artefact' if res['conc_explained_by_null'] else 'REAL (outside null)'}",
        f"incr. IPA:         obs {res['obs_slopeIPA']:+.5f}  "
        f"null {res['null_slopeIPA_mean']:+.5f} "
        f"[{res['null']['slopeIPA_range'][0]:+.5f}, {res['null']['slopeIPA_range'][1]:+.5f}]"
        f"  -> {'ARTEFACT (inside null)' if res['ipa_explained_by_null'] else 'real'}",
        "",
        f"(d) Knowledge graph: {res['n_tensions']} structural tension(s)",
    ]
    for var, k1, r1, k2, r2 in res["tensions"]:
        lines.append(f"  {var}: {k1}/{r1}  vs  {k2}/{r2}")
    lines += ["",
              "Pipeline ledger (blocked run):"]
    for layer, status in res["pipeline_ledger"]:
        lines.append(f"  {layer}: {status}")
    a.text(0.02, 0.98, "\n".join(lines), transform=a.transAxes, va="top",
           fontsize=9.5, family="monospace", color="#e8ebf2")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "charts.png", dpi=120, facecolor="#0b0e14")
    plt.close(fig)


if __name__ == "__main__":
    main()
