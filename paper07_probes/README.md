# Paper 7 probes — "Bayesian Validation of Cross-Domain Concept-Map Edges for Enterprise KPI Governance"

Sole author: Ranadhir Ghosh. This folder deposits the replication
package for the manuscript above, submitted to *Data & Knowledge
Engineering* on 2026-10-27 (Ms. No. DATAK-D-26-02293; desk-rejected
2026-11-04 on scope). The manuscript's Data and Code Availability
statement promised that these artifacts were "intended for public
deposit with the submission package" and that "the corresponding
author will provide the repository location at submission." That
deposit was not made at submission. It is made here, on 2026-11-05,
as step P7-E1 of the paper's rejection-loop record; the paper is
parked as a deposited, citable record. The corrected availability
statement is in `AVAILABILITY_STATEMENT.md` in this folder.

All code is deposited as run (unedited), with its recorded results.
One disclosed exception to direct re-runnability:
`probes/misspecification_probes.py` imports the benchmark module
through one absolute path from the author's workspace (line 16);
adjust that single path to your local copy of
`probes/original_benchmark_copy.py` before running it. Every other
script locates its inputs relative to its own position, in the
layout described below.

## Layout (mirrors the build record)

- `probes/` — the base procedure: benchmark, decision rule of
  record, bridge sampling, misspecification controls, field study,
  strengthening checks.
- `upgrade_probes/` — upgrade cycle 1: the covariate-adjusted
  confirmation stage (deterministic computation) and the expanded
  21-edge field ledger.
- `upgrade2_probes/` — upgrade cycle 2: the reliability-corrected
  stage, the ledger-level posterior-expected-FDP rule, the severity
  suite, and the large-ledger experiment.
- `tester_probes/`, `tester_probes_r2/`, `tester_probes_r3/`,
  `tester_probes_r4/` — the independent testers' own probes
  (independently coded reimplementations and re-runs), deposited as
  the verification record. These are the testers' artifacts, not the
  author's pipeline.
- `field_study_preregistration.md`, `upgrade_preregistration.md`,
  `upgrade2_preregistration.md` — the three pre-registrations, each
  committed before the runs it governs.
- `field_data/DATA_PROVENANCE.md` — the field datasets are public
  and are **not** redistributed here; the provenance file names each
  dataset, its source and licence, and the SHA-256 of the exact file
  used, so the field runs reproduce from the original downloads
  placed in `field_data/`.
- `upgrade_data/extracted/day.csv` — the extracted day-level file of
  the UCI Bike Sharing dataset (CC BY 4.0). The field code reads the
  original archive `bike_sharing.zip`; its SHA-256 is recorded in
  the provenance file.

## Contents and headline values

**Benchmark** — `probes/original_benchmark_copy.py`, results in
`probes/kpi_governance_results.json`. The pre-specified 19-hypothesis
benchmark with known ground-truth slopes: all 18 non-null edges
accepted, the embedded null rejected (Savage–Dickey BF01 = 134.4 in
the recorded run; the bridge-sampling analysis below removes the
display ceiling — the manuscript reports BF01 ≈ 305 at its reference
prior). Mean relative error of the posterior-mean slope 0.073 over
the non-null edges; 94% HDI covers the truth on 16 of 18.

**Decision rule of record** — `probes/rule_of_record.py`. The
published three-way rule (accept / reject / inconclusive from BF10
and the directional probability), with its unit tests in the same
file; running it prints "rule_of_record unit tests: ALL PASS".

**Bridge sampling** — `probes/bridge_rhat_multiseed.py`, results in
`probes/bridge_rhat_multiseed_results.json`. Log-domain bridge
sampling removes the Savage–Dickey display ceiling and is certified
against deterministic quadrature; multi-seed R-hat diagnostics.
`probes/bridge_debug.py` is the diagnostic script used while
developing it (no separate results file).

**Misspecification controls** — `probes/misspecification_probes.py`,
results in `probes/misspecification_results.json`. All-null battery
(the 95-run control battery reported in the manuscript produced zero
false ratifications), heavy-tailed noise, and omitted-nonlinearity
probes.

**Strengthening checks** — `probes/strengthening_checks.py`,
results in `probes/strengthening_checks_results.json`. Multi-seed
decision vectors under both the shipped and the published rules,
plus the common-cause confounder control.

**Field study** — `probes/field_study.py`, results in
`probes/field_study_results.json`, pre-registration in
`field_study_preregistration.md`. The frozen procedure run on public
datasets (Olist e-commerce, IBM Telco churn, UCI Bike Sharing).

**Upgrade cycle 1** — `upgrade_probes/adjusted_core.py`,
`run_simulation.py` (results `simulation_results.json`),
`run_field_upgrade.py` (results `field_upgrade_results.json`). The
covariate-adjusted confirmation stage, whose omitted-variable-bias
identity is proved in the manuscript; the adjusted computation is
deterministic and certified against bivariate quadrature. The
expanded field ledger covers 21 edges; 13 remain ratified after
adjustment.

**Upgrade cycle 2** — `upgrade2_probes/corrected_core.py`,
`probe_algebra.py` (results `algebra_results.json`: the corrected
population slope equals the true adjusted slope, verified to
4.4e-16), `run_severity2.py` (results `severity2_results.json`),
`run_large_ledger.py` (results `large_ledger_results.json`, log
`large_ledger.log`), `run_field2.py` (results
`field2_diagnostic_results.json`). The reliability-corrected stage's
grid-robustness rule restores the ledger false-discovery proportion
from 0.249 to 0.002 under proxy measurement without knowing the true
reliability; the posterior-expected-FDP ledger rule is reported as
demoted to "budget language" under the paper's pre-committed clause
(it removes only true edges where it binds: 174 true / 0 false in
the large-ledger experiment).

## Verification (deposit-time, 2026-11-05)

Before this deposit, every author-side script was re-run from a
clean copy in this layout (Python 3.12, NumPy 1.26.4, pandas 2.1.4,
SciPy 1.11.4), and each regenerated results file was compared
value-by-value against the recorded file deposited here. Outcome:
`kpi_governance_results.json`, `field_study_results.json`,
`misspecification_results.json`, `simulation_results.json`,
`field_upgrade_results.json`, `algebra_results.json`,
`large_ledger_results.json`, `severity2_results.json` and
`field2_diagnostic_results.json` regenerated **identically**;
`bridge_rhat_multiseed_results.json` and
`strengthening_checks_results.json` regenerated identically except
their recorded wall-time fields. The decision-rule unit tests pass.
The testers' probes were not re-run at deposit time; they stand as
the independent verification record described in the manuscript and
its audit trail.
