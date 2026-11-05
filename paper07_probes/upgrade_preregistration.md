# Paper 7 — UPGRADE pre-registration (written BEFORE any upgrade model is run on field data)

**Written:** 2026-10-27 15:01 EDT (19:01 UTC), prior to fitting any adjusted,
ledger-level, or simulation model of the upgrade cycle. Dataset documentation
and column names were inspected to specify the edges below; **no model has
been fitted to field data under this protocol at the time of writing.**
This document freezes the edge list, directions, covariate sets, subsample
rules, the stage-1 / stage-2 / ledger protocol, and the simulation spec.
Any deviation discovered during execution is reported in the upgrade build
record, never silently absorbed.

## 1. Datasets (sha256, as on disk 2026-10-27)

| File | sha256 | rows |
|---|---|---|
| field_data/olist_orders_dataset.csv | 8df58ef3d2d7e9944010f7beecd9b75367f5588ec6e3c91cec19ae3345ef9ecf | 99,441 |
| field_data/olist_order_reviews_dataset.csv | 012b61c7593e34f51fa614efdf802b9c7056ce6aae5307ddb93236e7cfc797d7 | 99,224 records (104,719 lines) |
| field_data/olist_order_payments_dataset.csv | 61674868ed7b872aac3dcd95bf774448a88f6426b183f21be55f75af021ffdec | 103,886 |
| field_data/olist_order_items_dataset.csv | 4f6abdbbc94036d0df4a76fa0520c072e31a40119d70f7f370fba1e2285d2bcb | 112,650 |
| field_data/olist_customers_dataset.csv | c26c17f59f3027a6a0dbeb8f1fa373c38f3323ecc84f225a8f4820e5bb288df8 | 99,441 |
| field_data/telco_churn.csv | 16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91 | 7,043 |
| upgrade_data/bike_sharing.zip (as downloaded) | b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401 | — |
| upgrade_data/extracted/day.csv (extracted from the zip above) | a6bcf826782d3c0fbfdcbeead17cd0884185a0dafe8ff10cd48a874ee7ba18be | 731 |

Sources: Olist public release and IBM Telco sample (as in
`field_study_preregistration.md`); UCI Bike Sharing (Fanaee-T & Gama),
`day.csv`, day-level records 2011–2012.

## 2. Frozen computation protocol (all stages, all field edges)

- **Standardisation.** For each edge, on its analysis sample, every variable
  (focal x, outcome y, each covariate) is z-scored with the sample mean and
  the population standard deviation (ddof = 0). All modelling is on the
  standardised scale.
- **Adjusted model (stage 2).** y = α + β·x + Σ_k γ_k·c_k + ε,
  ε ~ N(0, σ²). Priors, independent, on the standardised scale:
  α ~ N(0, 10²); β, γ_k ~ N(0, 5²); σ ~ HalfNormal(scale = max(sd(z_y), 1) = 1).
  **Null model:** identical except β ≡ 0; the covariates are present, with the
  same priors, in BOTH models. BF10_adj = p(y | M1) / p(y | M0).
- **Computation (deterministic; no MCMC).** Conditional on σ the coefficient
  posterior is Gaussian and the marginal likelihood p(y | σ) is analytic
  (p×p posterior-precision / Woodbury form). σ is integrated by trapezoid
  quadrature over log σ on a fixed grid: log σ ∈ [log 0.02, log 8],
  1,201 equally spaced points. The directional probability P_d_adj is the
  σ-mixture posterior probability that β has the elicited sign (posterior
  weights from the same quadrature; per-σ normal CDF of the focal slope).
  Adjusted posterior mean of β is the corresponding σ-mixture mean.
- **Stage 1 (bivariate).** The identical computation with an EMPTY covariate
  set, on the same analysis sample, giving BF10_biv and P_d_biv.
- **Decision rule (both stages).** The published rule of record
  (`probes/rule_of_record.py`): accept-strong BF10 ≥ 10 ∧ P_d ≥ 0.95;
  accept-moderate 3 ≤ BF10 < 10 ∧ P_d ≥ 0.90; reject BF10 ≤ 1/3; else
  inconclusive. **Stage-2 ratified** ⟺ stage-1 decision = accept AND
  stage-2 decision = accept.
- **Ledger rule (stage 3).** For each edge, with prior inclusion probability
  w (primary w = 0.5; sensitivity w = 0.1, 0.9 reported):
  odds = (w/(1−w))·BF10_adj; p_inc = odds/(1+odds);
  p_claim = p_inc · P_d_adj.
  Among stage-2 ratified edges, order by p_claim descending and ratify the
  largest prefix whose mean(1 − p_claim) ≤ γ, γ = 0.05. The ledger outcome
  is reported at all three w values.
- **Certification (frozen acceptance criterion).** With an empty covariate
  set the adjusted computation must reproduce the EXISTING bivariate
  quadrature (`probes/field_study.py: quad_log10_bf`, applied to the same
  standardised x, y) to within 0.01 log10 BF units on each of the 7 original
  field edges. The maximum absolute discrepancy is reported whatever it is.
- **Analysis samples.** Complete cases per edge (focal, outcome, AND all its
  covariates non-missing). Olist and Telco: if complete cases > 2,000, draw a
  uniform subsample of exactly 2,000 with seed 20261027 (pandas
  `sample(n=2000, random_state=20261027)`, the original field convention);
  otherwise use all. Bike Sharing: all 731 day-level cases, no subsample.
- **Reference column.** For the 7 original edges the previously published
  raw-scale stage-1 decisions (`probes/field_study_results.json`) are
  reported alongside for comparison only; they are not inputs to any
  upgrade decision.

## 3. Frozen field ledger (21 edges)

Constructions for edges 1–7 are exactly those of
`field_study_preregistration.md`. Derived variables: `freight_total` = sum of
freight_value per order; `items_value` = sum of price per order; `n_items` =
count of item rows per order; `delay_days` as before; `zip3` as before.
Telco binary variables (all covariates numeric/binary by construction):
`Partner01` = 1{Partner = Yes}; `Dependents01` = 1{Dependents = Yes};
`Paperless01` = 1{PaperlessBilling = Yes};
`M2M01` = 1{Contract = Month-to-month} — **the stated binary collapse of the
3-level Contract variable frozen here** (month-to-month vs any fixed term).
`churn01` = 1{Churn = Yes}. Olist `state_code`: customer_state mapped to its
rank in the alphabetically sorted list of distinct state codes present in
the customers file (a purely nominal numeric labelling, frozen as the
control's construction).

### Original edges (covariate sets new to this protocol)
1. `olist_delay_drives_review` — delay_days → review_score (−).
   Covariates: freight_total, items_value, n_items.
2. `olist_freight_drives_review` — freight_total → review_score (−).
   Covariates: delay_days, items_value, n_items.
3. `olist_installments_drive_value` — max payment_installments →
   total payment_value (+). Covariates: n_items, freight_total.
4. `olist_zip_null_control` — zip3 → review_score (+, implausible control).
   Covariates: delay_days, freight_total.
5. `telco_tenure_drives_charges` — tenure → MonthlyCharges (+).
   Covariates: SeniorCitizen, Partner01, M2M01.
6. `telco_tenure_drives_churn` — tenure → churn01 (−, linear-probability
   reading). Covariates: MonthlyCharges, SeniorCitizen, M2M01.
7. `telco_charges_drive_total` — MonthlyCharges → TotalCharges (+, positive
   control). Covariates: tenure, SeniorCitizen.

### New Olist edges
8. `olist_nitems_drive_freight` — n_items → freight_total (+).
   Covariates: items_value, delay_days.
9. `olist_freight_drives_delay` — freight_total → delay_days (+).
   Covariates: items_value, n_items.
10. `olist_review_drives_freight` — review_score → freight_total (+).
    **Deliberately implausible control:** the review is written after
    delivery, hence after the freight charge is fixed; a ratified edge
    here can only be association read backwards in time.
    Covariates: items_value, n_items.
11. `olist_statecode_drives_review` — state_code → review_score (+).
    **Deliberately implausible control:** an arbitrary nominal labelling
    cannot drive satisfaction. Covariates: delay_days, freight_total.
12. `olist_nitems_drive_review` — n_items → review_score (−).
    Covariates: delay_days, freight_total, items_value.

### New Telco edges
13. `telco_senior_drives_churn` — SeniorCitizen → churn01 (+).
    Covariates: tenure, MonthlyCharges, M2M01.
14. `telco_partner_drives_tenure` — Partner01 → tenure (+).
    Covariates: SeniorCitizen, Dependents01.
15. `telco_charges_drive_churn` — MonthlyCharges → churn01 (+).
    Covariates: tenure, SeniorCitizen, M2M01.
16. `telco_paperless_drives_churn` — Paperless01 → churn01 (+).
    Covariates: tenure, MonthlyCharges, M2M01.

### New Bike Sharing edges (day.csv, n = 731, all cases)
17. `bike_temp_drives_cnt` — temp → cnt (+).
    Covariates: hum, windspeed, workingday, yr.
18. `bike_hum_drives_cnt` — hum → cnt (−).
    Covariates: temp, windspeed, workingday, yr.
19. `bike_windspeed_drives_cnt` — windspeed → cnt (−).
    Covariates: temp, hum, workingday, yr.
20. `bike_workingday_drives_cnt` — workingday → cnt (+).
    Covariates: temp, hum, windspeed, yr.
21. `bike_yr_drives_cnt` — yr → cnt (+).
    Covariates: temp, hum, windspeed, workingday.

No further edges may be added after any upgrade model is run; edges may not
be dropped from the report for any reason.

## 4. Frozen simulation spec (ledger-level calibration)

- **Worlds:** 120 worlds, master seed 20261027 (NumPy default_rng). Each
  world is a ledger of m = 40 edges; n = 180 observations per edge.
- **Composition (fixed counts per world):** 16 true direct edges,
  12 pure nulls, 12 confounded-only edges.
- **Generating process (per edge; z, noises independent standard normal):**
  - True direct: covariate c ~ N(0,1); x ~ N(0,1) independent;
    β ~ Uniform(0.25, 0.60), elicited direction +1;
    γ ~ Uniform(−0.40, 0.40); y = β·x + γ·c + ε, ε ~ N(0,1).
    The measured covariate supplied to stage 2 is c.
  - Pure null: x ~ N(0,1), c ~ N(0,1) independent; y = γ·c + ε,
    γ ~ Uniform(−0.40, 0.40). Stage 2 receives c.
  - Confounded-only: common cause z ~ N(0,1);
    λ_x, λ_y ~ Uniform(0.50, 0.80) independently;
    x = λ_x·z + sqrt(1−λ_x²)·ε_x; y = λ_y·z + sqrt(1−λ_y²)·ε_y;
    direct slope exactly 0, elicited direction +1.
    The measured confounder supplied to stage 2 is z itself.
- **Fitting:** the frozen protocol of §2 (standardisation within each
  edge's sample; same priors, grid, and decision rule; stage 2 receives
  the one measured covariate/confounder named above).
- **Per-world outcomes, for each of three procedures** —
  (a) stage-1 bivariate rule alone; (b) two-stage adjusted rule;
  (c) two-stage + ledger-FDR prefix rule (γ = 0.05, w = 0.5):
  number ratified, number of FALSE ratifications (a ratified edge that is
  not a true direct edge), realised FDP = false / max(ratified, 1) with
  FDP := 0 when nothing is ratified, and number of true direct edges
  ratified. Confounded-only ratifications are also counted separately.
- **Reporting:** across the 120 worlds, mean / median / min / max of each
  outcome per procedure, taken from the saved simulation JSON. Every
  simulation number printed in the manuscript comes from that JSON.
- **Honest expectation (recorded in advance, not a success criterion):**
  (a) ratifies most confounded-only edges; (b) removes most of them;
  (c) holds realised FDP near or below γ on average. Whatever the JSON
  shows is what the manuscript reports, including disappointment.

## 5. What this protocol does not claim

Adjusted confirmation controls the MEASURED covariates listed above only.
Unmeasured confounding remains a threat to every field edge and is stated
as such wherever the upgrade is described. No ground truth exists for any
field edge and none will be claimed.
