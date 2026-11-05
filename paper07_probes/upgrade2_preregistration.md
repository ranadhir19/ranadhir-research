# Paper 7 — Upgrade Cycle 2 Pre-registration

**Frozen:** 2026-10-27 15:34 EDT (19:34 UTC), BEFORE any cycle-2 experiment
was run. Cycle-2 builder pass responding to the Round-3 hypothesis test
(`files/Paper_07_Hypothesis_Test_Report_R3.md`). No manuscript number in
cycle 2 may come from anywhere but the JSON outputs specified here.

## 0. References verified in-session before freezing (Crossref / publisher records)

- Fuller, W. A. (1987), *Measurement Error Models*, Wiley — DOI
  10.1002/9780470316665 (Crossref-asserted in multiple indexed records).
- Carroll, R. J., Ruppert, D., Stefanski, L. A., Crainiceanu, C. M. (2006),
  *Measurement Error in Nonlinear Models: A Modern Perspective*, 2nd ed.,
  Chapman & Hall/CRC — DOI 10.1201/9781420010138 (Crossref-asserted).
- Hagar, L., Stevens, N. T., "Design of Bayesian A/B Tests Controlling
  False Discovery Rates and Power", arXiv:2312.10814 (authors, title and
  abstract verified on the arXiv record opened in-session, v6, 2025-09-27;
  cited in later literature as Hagar & Stevens 2025, arXiv preprint).
- `bayefdr` R package: CRAN Task View (Bayesian Inference) states it
  "implements the Bayesian FDR control described by Newton et al. (2004)"
  (verified in-session on the CRAN Task View page).

## 1. Component C — corrected computation (`upgrade2_probes/corrected_core.py`)

Model/priors/grid identical to cycle-1 `adjusted_core.py` (standardised
variables, ddof=0; priors N(0,5^2) slopes, N(0,10^2) intercept,
half-normal(1) on sigma; log-sigma trapezoid grid 1201 points on
[log 0.02, log 8]).

**Correction (frozen definition).** For reliability lambda in (0,1], the
corrected Gram matrix is the observed Gram matrix (X'X on standardised
variables, X = [1, x, covariates]) with ONLY the diagonal entries of the
covariate columns multiplied by lambda. X'y and y'y are unchanged (under
classical error, Cov(z*, x) = Cov(z, x) and Cov(z*, y) = Cov(z, y)).
The null model retains the covariates with the same correction. The
corrected marginal likelihoods, BF(lambda), focal posterior mean and
directional probability P_d(lambda) are then computed by the same
analytic-coefficient-integration + log-sigma quadrature formulas as
cycle 1, applied to the corrected Gram. This is labelled everywhere a
*method-of-moments corrected Bayes factor* (approximation to the EIV
posterior, NOT the exact latent-z posterior).

**Multi-covariate application (field/severity generalisation, frozen):**
the correction is applied simultaneously to every covariate column's
diagonal (common lambda). The single-proxy case is the proposition's
case and is the severity suite's V3 design (one covariate).

**Positive-definiteness fallback (frozen):** if the corrected posterior
precision fails Cholesky at any grid sigma for a given lambda, that
(edge, lambda) is recorded `not_evaluable` and counts as NOT accept for
all decision/robustness purposes.

**Robustness rule (frozen):** grid Lambda = {1.0, 0.9, 0.8, 0.7, 0.6}.
lambda_min(edge) = smallest grid lambda such that the stage-2 corrected
decision is ACCEPT at that lambda and at every larger grid lambda;
"—" if the decision is not accept at lambda = 1.0 (or not evaluable
at 1.0). An edge is reliability-robust ratified iff it is stage-1
accept AND lambda_min = 0.6 evaluated contiguously (i.e. accept at
every grid lambda) — reported via procedure (e) in V3.

## 2. Algebra verification (`upgrade2_probes/probe_algebra.py` -> `algebra_results.json`)

- 500 random population settings, seed 2026102702: a = Corr-loadings
  drawn U(-0.85, 0.85), beta U(-0.8, 0.8), gamma U(-1, 1), residual sd
  U(0.5, 1.5), lambda in {0.5, 0.64, 0.8, 0.9, 1.0} cycled. For each:
  exact analytic population covariance of (x, z, y) and of (x, z*, y)
  (z* = z + u, Var(u) = Var(z)(1-lambda)/lambda), standardised;
  corrected slope from Sigma(lambda) vs true adjusted slope from the
  true covariance. Report max |discrepancy|.
- Empirical brute force: 3 fixed settings, n = 1,000,000 draws,
  seed 2026102703; sample covariances; corrected slope vs OLS adjusted
  slope using the true z on the same sample. Report discrepancies.

## 3. Lambda = 1 certification (from `run_field2.py` output)

On all 21 field edges: max |log10 BF_corr(1.0) - adj_log10_BF10| against
cycle-1 `field_upgrade_results.json`. Criterion: <= 0.01 log10 units
(at least 4 edges required; all 21 will be reported).

## 4. Severity suite (`upgrade2_probes/run_severity2.py` -> `severity2_results.json`)

Spec = tester R3 `probe_r3_sim.py` variants, re-implemented with cycle-1
builder code (`adjusted_core.fit_adjusted` + `rule_of_record.decide`).
World composition: 40 edges = 16 true / 12 pure null / 12 confounded-only;
direction +1 for all edges; per-edge generation EXACTLY as the tester
code (true: y = beta x + gamma c + eps, gamma U(-0.40,0.40); null:
y = gamma c + eps; confounded: x = lx z + sqrt(1-lx^2) ex,
y = ly z + sqrt(1-ly^2) ey, loadings U(0.50,0.80) unless variant says
otherwise; covariate supplied to stage 2 = c resp. z unless variant
says otherwise). **30 worlds per variant.** Master seeds (frozen):
V1 20261101, V2 20261102, V3 20261103, V4 20261104, V5 20261105,
V6 20261106. Variants (one change each from baseline n=180,
beta U(0.25,0.60), loadings U(0.50,0.80)):
V1 loadings U(0.25,0.45); V2 n=60; V3 covariate = z + 0.75*N(0,1)
(proxy noise sd 0.75; reliability lambda_true = 1/1.5625 = 0.64);
V4 covariate = independent N(0,1) noise; V5 beta U(0.10,0.30);
V6 true edges only: x = 0.7*c + sqrt(1-0.49)*N(0,1).

Procedures: (a) bivariate (stage-1 decision alone); (b) two-stage
(stage-1 accept AND stage-2 accept). V3 additionally:
(d) ORACLE two-stage with reliability correction at the TRUE
lambda = 0.64 (stage-1 accept AND corrected stage-2 accept at 0.64);
(e) robustness rule (stage-1 accept AND corrected stage-2 accept at
every lambda in Lambda; lambda=1.0 cell reuses procedure (b)'s stage-2
fit — justified only if the Section-3 certification passes; the
certification is computed on field edges, and corrected_core at
lambda=1.0 is algebraically the uncorrected Gram).
Report per variant/procedure: mean ratified, mean false/world, mean
FDP, mean true retained (of 16). V3 also: lambda_min distribution over
edges, (d)/(e) as above.

## 5. Large-ledger experiment (`upgrade2_probes/run_large_ledger.py` -> `large_ledger_results.json`)

**40 worlds, m = 120 edges/world, master seed 20261201 (frozen).**
Composition per world: 48 true (40%), 36 pure null (30%), 36
confounded-only (30%). Per-edge n drawn i.i.d. from {60, 120, 180}
(rng choice, per edge, before edge generation). True edges:
beta U(0.15, 0.60) (deliberately weaker than cycle 1), gamma
U(-0.40, 0.40), covariate = c measured exactly. Nulls: gamma
U(-0.40, 0.40), covariate = c. Confounded-only: loadings
U(0.50, 0.80), covariate = z measured exactly. Direction +1.
Procedures: (a) bivariate; (b) two-stage; (c) two-stage + ledger-FDR
prefix rule (gamma = 0.05) at w = 0.5 (procedure c) and separately at
w = 0.1 (procedure c01), p_claim from stage-2 adjusted BF/Pd.
Report per procedure: mean ratified / false / true retained (of 48) /
FDP. Decisive table: number of worlds where (c) != (b) at each w;
total edges removed by (c) across worlds, split false vs true.
Calibration: per world, posterior-expected FDP of the (b) set
(mean(1 - p_claim) at w = 0.5 over (b)-ratified edges) vs realised
FDP of the (b) set; report both means and the mean difference.
Outcome printed whatever it is (pre-committed demotion clause: if (c)
again removes only true edges, Component B is demoted in the
manuscript to budget-language-only with the confirmation stated).

## 6. Field diagnostic (`upgrade2_probes/run_field2.py` -> `field2_diagnostic_results.json`)

The 21 cycle-1 field edges, constructions imported unchanged from
cycle-1 `run_field_upgrade.py`. For each edge: corrected stage-2 fits
at every lambda in Lambda (correction applied to all covariates of
that edge simultaneously), recording log10 BF(lambda), Pd(lambda),
stage-2 decision, evaluability, and lambda_min (Section 1 definition,
on the stage-2 decision). DIAGNOSTIC ONLY: no field decision changes
on lambda grounds; conditional wording mandatory in the manuscript.

## 7. Text fixes R1–R5 (from R3 report Section 6)

Applied verbatim in substance, with tester numbers cited as
tester-found for R1/R2 (V3: FDP 0.26, 5.1 false/world; power 9.4/16
at rho=0.7, 6.0/16 at n=60, 3.9/16 at beta in [0.10,0.30]) and the
paper's own severity suite (Section 4) reported alongside.
R3: both-stages suppression trade-off sentence (bike_hum).
R4: Hagar & Stevens + bayefdr added to the ledger-FDR lineage.
R5: "retained, not filtered" budget language for the field ledger.
