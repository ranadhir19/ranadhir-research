# Horizon Durability of ECG Risk (cardio)

Original implementation of the core computational ideas from
**"Preliminary investigation: forecasting cardiovascular risk from the resting
ECG"** (Dr. Ranadhir Ghosh, shared with Dr. Debabrata Mukhopadhyay, Mayo
Clinic, Sep 2026). No text or code is taken from the paper; this is an
independent, runnable rendering of its measurement protocol. All experiments
run on **synthetic** constant-effect proportional-hazards data (n=4000) --
there is no cohort access here. The paper's full text is preserved verbatim at
`cardio/paper/PRELIMINARY_INVESTIGATION.html` (the author's own work, supplied
for this purpose).

## The paper

Can a resting ECG forecast cardiovascular risk years ahead -- and does that
information survive as the prediction window lengthens, or fade? A model that
scores well at one year and poorly at seven is detecting present disease; one
that holds its value is forecasting. The paper's contribution is not another
architecture but a measurement protocol that makes horizon claims falsifiable:
a direct Grambsch-Therneau test for time-varying coefficients, a
fixed-cohort landmark design, censoring-weighted statistics (Uno's
concordance, cumulative/dynamic AUC, IPCW Brier/IPA), a constant-effect null
simulation calibrating every horizon trend, an interaction-information screen
gating fusion-architecture investment, and a typed knowledge graph whose
structural tensions flag invariance-vs-trend contradictions.

## Core contribution implemented

`ghosh_research.cardio` on a synthetic cohort (age, sex, latent risk L;
redundant ECG views `ecg1`, `ecg2`, `annot` all functions of L; exponential
baseline; ~25% events):

- Cox PH by exact Newton-Raphson on the Breslow partial likelihood;
  Schoenfeld residuals; Grambsch-Therneau per-covariate and global tests.
- Uno's censoring-weighted concordance, cumulative/dynamic time-dependent AUC,
  IPCW Brier score and IPA, all truncated at horizon tau on a fixed landmark
  cohort (outcome known at 7.5y).
- Constant-effect null: 5 replicates regenerated with the same constant true
  betas; durability slopes (per year) of incremental concordance and
  incremental IPA compared against the null range.
- Interaction information `II(A;B) = I(A,B) - I(A) - I(B)` over modality
  pairs, then concatenation vs a tiny gated-fusion network (per-modality tanh
  encoders, softmax attention, 200 Adam steps on the partial likelihood).
- `ClaimGraph` (theorem/measured/assumption claims) with structural tension
  detection; `GatedPipeline` where a failed gate blocks the run.

## How to run

```bash
cd ~/workspace/research
PYTHONPATH=src python3 experiments/cardio_experiment.py   # full experiment
PYTHONPATH=src python3 examples/11_cardio_quickstart.py   # 1-second demo
```

The experiment writes `results/cardio/results.md` and `results/cardio/charts.png`.

## Results

Measured (synthetic n=4000, seed 0; null seeds 101-105):

| finding | measured |
|---|---|
| GT time-variation p (age, sex, ecg1, ecg2, annot) | 0.839, 0.262, 0.068, 0.927, 0.976 (all > 0.05); global 0.416 |
| incremental concordance slope/yr (landmark) | -0.00046, null range [+0.00036, +0.00252] -> **not** explained by the null |
| incremental IPA slope/yr (landmark) | +0.00898, null range [+0.00733, +0.01694] -> explained: **artefact** |
| II(demo;ecg), II(ecg1;ecg2), II(demo;annot) | -367.6, -552.6, -248.5 nats (all redundant) |
| concat vs gated Uno C (tau=7.5) | 0.7408 vs 0.7358 (concat wins by +0.0049) |
| knowledge-graph tensions | 2 (invariance vs trend on C_tau and on ecg_effect) |

The rising incremental proper score is reproduced by the constant-effect null:
it is what the statistic does when nothing is changing. The falling
incremental concordance lies outside the null range: it is not the instrument.

## Insights

- The Grambsch-Therneau test is the direct question -- a slope through
  horizon-windowed discrimination is indirect and spacing-dependent. Here it
  finds no time variation while the truncated concordance still drifts: the
  tension the paper leaves standing.
- A proper score (IPA) cannot be improved by miscalibration, which is why its
  rise under the null is diagnostic of the *statistic*, not the model. Report
  a horizon trend with no null reference and you will report artefacts.
- Interaction information is signed, which a concordance difference is not --
  and the sign is the architectural decision. Negative everywhere here
  predicts, before any network is written, that gated fusion loses. It did.
- A gate that does not stop the run is a comment. The synergy screen's
  refusal to build the fusion network *is* the pipeline working.

## Implementation notes (honest deviations)

- The paper's falling incremental concordance is unexplained in the real data;
  a pure constant-effect miniature cannot produce it (its slope would sit
  inside the null range like the IPA's). This implementation therefore gives
  the observed cohort one concrete candidate mechanism -- risk-correlated ECG
  measurement noise (`het_k=2.0`: noisier views for low-risk patients) -- with
  true coefficients held constant so the GT test stays blind to it. The null
  replicates are the pure homoscedastic constant-effect world.
- Magnitudes are miniature-scale (slopes ~1e-3/yr, II in the hundreds of nats
  at n=4000) and should not be compared numerically with the paper's cohort
  figures; the qualitative pattern (GT null, IPA artefact, concordance
  unexplained, II negative, concat beats gated, tensions fire) is what is
  reproduced.
- The PCA dimension-vs-label-budget demo (9-dim quadratic ECG expansion,
  budgets 400/1500/4000) shows gains growing with dims at every budget -- with
  only 9 dims the paper's optimum-inversion does not appear; it is reported
  as measured.
