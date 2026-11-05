# Paper 2 probes — "The Agent Closure Theorem"

Sole author: Ranadhir Ghosh. This folder deposits the labelled
probes whose numerical values are reported in the manuscript "The
Agent Closure Theorem: An α-Budget on Certified Novelty in
Self-Referential AI Systems". Probes 4 and 6 were described in the
manuscript's Data Availability statement as "reported in the
accompanying build documentation" until their deposit (elevation
cycle E5); they are the author's build artifacts, deposited verbatim
(the code that produced the reported numbers, unedited). Probe 7
was added in elevation cycle E6 (2026-11-05); its provenance is
stated in its own section below.

## Probe 4 — sequential α-budget probe (manuscript §"The sequential
budget" and Appendix, "Numerical probe of the sequential regimes")

- `probe4_sequential.py` — the probe code. Deterministic: all seeds
  are `zlib.crc32` of fixed string labels; no other randomness.
  Requires Python 3 and NumPy.
- `probe4_output.txt` — the recorded output of the run the
  manuscript reports.

Reported values: predictable history-adaptive policy validates at
empirical rate 0.0485 (mean budget 0.0485, cap 0.05); the lifetime
schedule B_t = 0.5·(6/π²)/t² spends cumulative budget 0.5000 over
50,000 steps and total validations across 2,000 runs average 0.5015
(Monte Carlo standard error 0.014, per-run standard deviation
0.6334; maximum 3), with fraction 0.4325 of runs validating at
least once; the attaining construction's empirical rate is 0.0347
against mean budget 0.0350; the candidate-constructed
counterexample validates at rate 1.0000 at nominal α = 0.05 and at
nominal size 0, against 0.0497 and 0.0000 for the same tests fixed
ex ante.

## Probe 6 — instantiation audit (manuscript §"Instantiation: an
audit of a real certification pipeline")

- `probe6_instantiation.py` — the probe code. It audits the
  enriched-observability pipeline of the public `agentic-systems-lab`
  repository (system 16). To run it, check out
  https://github.com/ranadhir19/agentic-systems-lab and point the
  `sys.path` line at the top of the file at your checkout's `src`
  directory (as deposited, it names the author's local checkout
  path). Requires Python 3 and NumPy. Deterministic: it uses only
  the pipeline's own seeded generators (calibration seeds 7, 11;
  audit seeds 42, 0, 99 — the pipeline harness's seed list, split
  in list order), the pipeline's shipped threshold |z| ≥ 3.0, and
  no randomness of its own.
- `probe6_output.txt` — the recorded output of the run the
  manuscript reports (per-check α̂_j with Wilson 95% intervals,
  per-layer budgets B̂, the sequential audit, and the three
  predictability regimes).
- `probe6_records.csv` — the candidate-level records of that run:
  11,250 rows (one per seed × layer × minute from minute 30), with
  columns seed, split, layer, minute, union_clean, flags_per_check,
  union_flag, first_check_flag, selected_check_max_absz,
  selected_absz.

### Audit specification (as run)

Each check is the pipeline's shipped detector for one metric:
`enrich.robust_z` of the public `agentic-systems-lab` repository —
a robust trailing-median/MAD z-score with **trailing window 60
minutes** (the window at minute t covers the up-to-60 minutes
before t) and **MAD scaling 0.6745**, i.e. z = 0.6745 · (x −
median) / MAD over the trailing window (MAD floored by 1e-9 in the
implementation). Certification is per candidate minute at the
shipped threshold |z| ≥ 3.0, using the per-minute flag underlying
the pipeline's interval detector **without** its five-minute
interval grouping (min_len = 5 not applied), and with the first
30 minutes of each stream treated as warmup (no flags counted).
A minute is clean for a check if it lies outside every
ground-truth anomaly interval the pipeline's seeded generator
recorded for that metric, and clean for its layer if clean for
all of the layer's metrics.

## Probe 7 — labelled two-regime probe (manuscript Appendix A,
"Flags are not discoveries")

- `probe7_labelled.py` — the probe code: one run of 200,000 steps
  per agent on the manuscript's testbed (implementation B's
  `run_agent` draw sequence, module `ghosh_research.closure`),
  with the ground-truth regime label of every step recorded.
  Deterministic: seed 12345 for both agents. Requires Python 3,
  NumPy and SciPy.
- `probe7_output.txt` — the recorded output of the run the
  manuscript reports.

Reported values: hybrid agent — 65,866 flags over 200,000 steps
(rate 0.3293), of which 58,764 genuinely novel and 7,102
known-regime false alarms; precision 0.8922; recall 0.9802 over
59,950 novel-regime steps. Pure agent — 9,991 flags (rate 0.0500),
of which zero genuinely novel.

**Provenance (disclosed).** The manuscript's original labelled
probe (the D-40 build probes, 200,000 steps, seed 12345) no longer
exists in the author's workspace, and its counts were not
reproducible from any deposited artifact. In elevation cycle E6
the figures were re-derived from implementation B — the deposited
testbed above — in this probe, and the manuscript reports this
probe's counts verbatim. The qualitative findings are unchanged:
precision ≈ 0.89 (about one flag in nine a known-regime false
alarm) and zero genuinely novel pure-agent flags.

## The simulation implementations (Appendix, "A simulation")

For completeness, the manuscript's two simulation implementations
are: **implementation A** — `hybrid_discovery_simulation.py` with
results `hybrid_discovery_results.json`, in the journal repository
(https://github.com/ranadhir19/journal), folder
`03_Agent_Closure_Theorem/experiments/`;
**implementation B** — this repository's closure application:
`experiments/closure_experiment.py` (module
`ghosh_research.closure`), with results in `results/closure/`.
Implementation B's recorded values (pure 0.0483 ± 0.0019; hybrid
0.3260 ± 0.0034) are the manuscript's implementation-B rows.
