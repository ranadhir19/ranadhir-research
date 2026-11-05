# Paper 7 — Field study pre-registration (written BEFORE any field analysis)

**Written:** 2026-10-27, prior to running any model on the field data.
**Purpose:** the strengthening item 1 of the round-1 Hypothesis Test Report — run the
frozen validation procedure on real KPI data, with no ground truth, and report the
decisions. This document freezes the edge list, directions, data rules, and pipeline
settings in advance; any deviation discovered during execution is reported in the
results file, not silently absorbed.

## Datasets (as downloaded 2026-10-27; sha256)

| File | sha256 | rows |
|---|---|---|
| olist_orders_dataset.csv | 8df58ef3d2d7e9944010f7beecd9b75367f5588ec6e3c91cec19ae3345ef9ecf | 99,441 |
| olist_order_reviews_dataset.csv | 012b61c7593e34f51fa614efdf802b9c7056ce6aae5307ddb93236e7cfc797d7 | 104,719 |
| olist_order_payments_dataset.csv | 61674868ed7b872aac3dcd95bf774448a88f6426b183f21be55f75af021ffdec | 103,886 |
| olist_order_items_dataset.csv | 4f6abdbbc94036d0df4a76fa0520c072e31a40119d70f7f370fba1e2285d2bcb | 112,650 |
| olist_customers_dataset.csv | c26c17f59f3027a6a0dbeb8f1fa373c38f3323ecc84f225a8f4820e5bb288df8 | 99,441 |
| telco_churn.csv (IBM Telco Customer Churn) | 16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91 | 7,043 |

Sources: Olist public release via github.com/olist/work-at-olist-data;
IBM Telco Customer Churn via a public mirror of the IBM sample file.

## Pre-registered HypothesisSpecs (7 edges)

Unit of analysis: one order (Olist) / one customer (Telco). Complete-case per edge
(rows missing either variable are dropped; the drop count is reported).
Subsample rule (frozen): if an edge has more than 2,000 complete cases, draw a
uniform random subsample of exactly 2,000 with seed 20261027; otherwise use all.

### Olist (delivery / satisfaction / payments)
1. `olist_delay_drives_review` — x = delivery delay in days
   (order_delivered_customer_date − order_estimated_delivery_date, delivered orders
   only); y = review_score; elicited direction **−1** (later delivery, worse score).
2. `olist_freight_drives_review` — x = total freight_value per order (items summed);
   y = review_score; direction **−1**.
3. `olist_installments_drive_value` — x = max payment_installments per order;
   y = total payment_value per order; direction **+1**.
4. `olist_zip_null_control` — x = customer zip-code prefix (first 3 digits, numeric);
   y = review_score; direction **+1**. *Deliberately implausible edge (real-data
   null-analogue): the honest outcome is reject or inconclusive.*

### IBM Telco
5. `telco_tenure_drives_charges` — x = tenure (months); y = MonthlyCharges;
   direction **+1**.
6. `telco_tenure_drives_churn` — x = tenure; y = churn coded 0/1 (linear-probability
   reading, stated as such); direction **−1**.
7. `telco_charges_drive_total` — x = MonthlyCharges; y = TotalCharges (blank
   TotalCharges rows dropped); direction **+1**. *Positive control: domain experts
   are confident this linkage is strong (TotalCharges accumulates MonthlyCharges).*

## Frozen pipeline (identical to the benchmark, rule of record)

- Model M1: y = α + β·(x − x̄) + ε; priors β ~ N(0, 5²), α ~ N(ȳ, 10²),
  σ ~ HalfNormal(0, max(sd(y),1)); M0: β = 0.
- Sampler: shipped Metropolis–Hastings settings — per-edge seed chain
  (base seed 2000 + index), 2 chains may be used for R̂ via the 4-chain protocol of
  the benchmark study; B = 2,000 burn-in, D = 4,000 draws; acceptance adaptation
  toward the [0.234, 0.50] band during burn-in.
- Evidence: bridge sampling (Meng & Wong / Gronau) log₁₀ BF₁₀, certified against the
  analytic-α quadrature on every field edge (report max |bridge − quad|).
- Decision rule — **the published rule of record** (manuscript Table 1):
  accept-strong BF₁₀ ≥ 10 ∧ P_d ≥ 0.95; accept-moderate 3 ≤ BF₁₀ < 10 ∧ P_d ≥ 0.90;
  reject BF₁₀ ≤ 1/3; else inconclusive. Post-hoc flag (annotation only, not a
  decision): "direction contradicted" when BF₁₀ ≥ 10 ∧ P_d ≤ 0.05.
- Report per edge: n, dropped, β̂ posterior mean, 94% HDI, P_d, bridge log₁₀BF₁₀,
  quadrature log₁₀BF₁₀, R̂, mean acceptance, decision.
- **No ground truth exists and none will be claimed.** Face-validity commentary, if
  any, is labelled as expectation-vs-outcome, never as accuracy.
