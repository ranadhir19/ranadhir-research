"""TESTER Probe A (independent) for Paper 7.

A1. Re-run the released replication benchmark from source and diff every
    summary statistic against the shipped kpi_governance_results.json.
A2. INDEPENDENT exact marginal likelihoods: integrate alpha and beta
    ANALYTICALLY (2x2 Gaussian formula) and sigma by fine 1-D quadrature
    over log sigma. This shares no code path with the builder's bridge
    sampler NOR with the builder's quadrature (which grids beta
    numerically). Subset of edges: itsm(0), uptime(9), nps(13), pmo(16),
    null(18). Data regenerated with the released generator's exact RNG
    sequence (RandomState(seed=i); uniform, then normal).
A3. Recompute multiseed summary statistics from the builder's JSON and
    compare against the manuscript's prose numbers.
"""
import importlib.util, json, sys
import numpy as np

SRC = "/home/hatch/workspace/github-fis-watch/mirrors-mcp/journal/07_Bayesian_KPI_Governance/experiments/kpi_governance_benchmark.py"
SHIP = "/home/hatch/workspace/github-fis-watch/mirrors-mcp/journal/07_Bayesian_KPI_Governance/experiments/kpi_governance_results.json"
BRIDGE = "/home/hatch/workspace/goals/10-journal-papers-submitted-to-good-journals/hidden_files/paper07_work/probes/bridge_rhat_multiseed_results.json"
OUT = "/home/hatch/workspace/goals/10-journal-papers-submitted-to-good-journals/hidden_files/paper07_work/tester_probes/probeA_results.json"

spec = importlib.util.spec_from_file_location("bench", SRC)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
shipped = json.load(open(SHIP))

# ---------- A1: fresh re-run, per-edge diff ----------
edges = []
for i, (slug, bs, n, xlo, xhi, sig, d) in enumerate(bench.BENCH):
    r = bench.run_edge(bs, n, xlo, xhi, sig, d, seed=i)
    rel = None if bs == 0 else abs((r["post_mean"] - bs) / bs)
    r.update(slug=slug, beta_star=bs, rel_err=rel,
             hdi_covers=bool(r["hdi_lo"] <= bs <= r["hdi_hi"]))
    edges.append(r)
nonnull = [e for e in edges if e["beta_star"] != 0]
rels = [e["rel_err"] for e in nonnull]
mine = dict(
    rel_err_mean=float(np.mean(rels)), rel_err_median=float(np.median(rels)),
    n_rel_err_le_9pct=int(sum(r <= 0.09 for r in rels)),
    n_hdi_covers_nonnull=int(sum(e["hdi_covers"] for e in nonnull)),
    n_bf_saturated=int(sum(e["bf10_saturated"] for e in edges)),
    accept_rate_mean=float(np.mean([e["accept_rate"] for e in edges])),
    null_bf01=float(1.0 / [e["bf10"] for e in edges if e["beta_star"] == 0][0]),
    decisions={e["slug"]: e["decision"] for e in edges},
)
s = shipped["summary"]
a1 = dict(
    shipped=dict(rel_err_mean=s["rel_err_mean"], rel_err_median=s["rel_err_median"],
                 n_rel_err_le_9pct=s["n_rel_err_le_9pct"], n_hdi_covers=s["n_hdi_covers"],
                 n_bf_saturated=s["n_bf_saturated"], accept_rate_mean=s["accept_rate_mean"],
                 null_bf01=s["null_edge_bf01"]),
    tester_rerun={k: mine[k] for k in ("rel_err_mean", "rel_err_median", "n_rel_err_le_9pct",
                                       "n_hdi_covers_nonnull", "n_bf_saturated",
                                       "accept_rate_mean", "null_bf01")},
    max_abs_postmean_diff_vs_shipped=float(max(
        abs(m["post_mean"] - sh["post_mean"]) for m, sh in zip(edges, shipped["edges"]))),
    decisions_match_shipped=all(mine["decisions"][sh["slug"]] == sh["decision"]
                                for sh in shipped["edges"]),
)
print("A1 shipped:", a1["shipped"])
print("A1 tester :", a1["tester_rerun"])
print("A1 max|d post_mean| =", a1["max_abs_postmean_diff_vs_shipped"],
      " decisions match:", a1["decisions_match_shipped"])

# ---------- A2: exact marginal likelihoods (alpha, beta analytic) ----------
PA, PS, SB = 10.0, 5.0, 5.0

def logml_given_sigma(X, y, sigma, prior_sd):
    """log N(y; 0, sigma^2 I + X diag(prior_sd^2) X') via 2x2 lemma."""
    n = len(y)
    S0inv = np.diag(1.0 / np.array(prior_sd) ** 2)
    A = S0inv + X.T @ X / sigma**2
    b = X.T @ y / sigma**2
    sign, logdetA = np.linalg.slogdet(A)
    logdetS0 = float(np.sum(np.log(np.array(prior_sd) ** 2)))
    quad = y @ y / sigma**2 - b @ np.linalg.solve(A, b)
    return (-0.5 * n * np.log(2 * np.pi) - n * np.log(sigma)
            - 0.5 * logdetS0 - 0.5 * logdetA - 0.5 * quad)

def logml_exact(x, y, model):
    xc = x - x.mean()
    if model == "m1":
        X = np.column_stack([np.ones_like(xc), xc]); psd = [PA, SB]
    else:
        X = np.ones((len(x), 1)); psd = [PA]
    ls = np.linspace(np.log(0.02), np.log(60.0), 6000)
    sg = np.exp(ls)
    vals = np.array([logml_given_sigma(X, y, s, psd)
                     - 0.5 * (s / PS) ** 2 - np.log(PS) - 0.5 * np.log(np.pi / 2)
                     + l for s, l in zip(sg, ls)])  # +ls: log-sigma Jacobian
    m = vals.max()
    return float(m + np.log(np.trapz(np.exp(vals - m), ls)))

bridge = {e["slug"]: e for e in json.load(open(BRIDGE))["edges"]}
a2 = {}
for idx in (0, 9, 13, 16, 18):
    slug, bs, n, xlo, xhi, sig, d = bench.BENCH[idx]
    rng = np.random.RandomState(idx)
    x = rng.uniform(xlo, xhi, n)
    y = 2.0 + bs * (x - x.mean()) + rng.normal(0, sig, n)
    exact = (logml_exact(x, y, "m1") - logml_exact(x, y, "m0")) / np.log(10)
    a2[slug] = dict(tester_exact_log10_bf10=exact,
                    builder_bridge_log10_bf10=bridge[slug]["bridge_log10_bf10"],
                    builder_quad_log10_bf10=bridge[slug]["quad_log10_bf10"],
                    tester_minus_bridge=float(exact - bridge[slug]["bridge_log10_bf10"]),
                    tester_minus_quad=float(exact - bridge[slug]["quad_log10_bf10"]))
    print(f"A2 {slug:32s} exact={exact:9.3f} bridge={bridge[slug]['bridge_log10_bf10']:9.3f} "
          f"quad={bridge[slug]['quad_log10_bf10']:9.3f} d_bridge={a2[slug]['tester_minus_bridge']:+.4f}")

# ---------- A3: multiseed prose audit ----------
bj = json.load(open(BRIDGE))
ms = bj["multiseed"]
a3 = dict(
    hdi_covers_per_seed=[m["n_hdi_covers"] for m in ms],
    hdi_mean_recomputed=float(np.mean([m["n_hdi_covers"] for m in ms])),
    manuscript_says_hdi_mean=15.6,
    mean_rel_err_range=[float(np.min([m["mean_rel_err"] for m in ms])),
                        float(np.max([m["mean_rel_err"] for m in ms]))],
    within9_range=[int(np.min([m["n_within_9pct"] for m in ms])),
                   int(np.max([m["n_within_9pct"] for m in ms]))],
    null_bf01_range=[float(np.min([m["null_bf01"] for m in ms])),
                     float(np.max([m["null_bf01"] for m in ms]))],
    decision_vectors_identical=len({tuple(sorted(m["decisions"].items())) for m in ms}) == 1,
    max_bridge_quad_absdiff=float(max(abs(e["bridge_minus_quad_log10"]) for e in bj["edges"])),
    prior_sens=bj["prior_sensitivity_null"],
)
print("A3:", json.dumps(a3, indent=1))
json.dump(dict(A1=a1, A2=a2, A3=a3), open(OUT, "w"), indent=2)
print("wrote", OUT)
