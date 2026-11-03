# CARDIO experiment results

Synthetic constant-effect proportional-hazards cohort, n=4000, seed=0.
Censoring U(0.5, 8.5) years; event rate 0.256;
landmark cohort (outcome known at 7.5y) n=1339.
Observed cohort uses risk-correlated ECG-view noise (`het_k=2.0`):
the miniature's concrete mechanism for the paper's unexplained concordance
drift (true coefficients stay constant, so the GT test stays blind to it).
Null replicates: homoscedastic, same constant true betas, seeds [101, 102, 103, 104, 105].

## Cox fit (full model, one fit; Newton-Raphson converged: True)

| covariate | log-HR |
|---|---|
| age | 0.0594 |
| sex | 0.3283 |
| ecg1 | 0.3084 |
| ecg2 | 0.1220 |
| annot | 0.0360 |

## (a) Grambsch-Therneau test of proportional hazards

1025 events inform the test. H0: coefficient constant in time.

| covariate | p (time variation) | verdict |
|---|---|---|
| age | 0.8385 | pass (>0.05) |
| sex | 0.2618 | pass (>0.05) |
| ecg1 | 0.0680 | pass (>0.05) |
| ecg2 | 0.9272 | pass (>0.05) |
| annot | 0.9758 | pass (>0.05) |

Global test p = 0.4160. No detectable time variation --
the ECG effect does not measurably decay, mirroring the paper (p=0.9403).

## (b) Horizon durability with constant-effect null calibration

One model fit per arm (demo-only vs demo+ecg); horizons obtained by
restricting the evaluation window on the fixed landmark cohort, never by
refitting. Incremental = (demo+ecg) - (demo-only).

| tau (y) | incr. Uno C | incr. tdAUC | incr. IPA |
|---|---|---|---|
| 2 | +0.0480 | +0.0486 | +0.0679 |
| 3 | +0.0417 | +0.0354 | +0.0700 |
| 4 | +0.0417 | +0.0336 | +0.0874 |
| 5 | +0.0415 | +0.0255 | +0.0826 |
| 6 | +0.0445 | +0.0235 | +0.1118 |
| 7 | +0.0431 | +0.0124 | +0.1066 |

### Durability slopes (per year) vs the constant-effect null

| measure (landmark) | observed slope | null slope (constant effect) | null range | explained by the metric? |
|---|---|---|---|---|
| incremental concordance | -0.00046 | +0.00170 | [+0.00036, +0.00252] | no |
| incremental tdAUC | -0.00642 | -- | -- | -- |
| incremental IPA | +0.00898 | +0.01104 | [+0.00733, +0.01694] | YES - fully |

Null slopes: [0.002, 0.00036, 0.00127, 0.00252, 0.00233] (concordance), [0.01245, 0.0107, 0.00733, 0.01694, 0.00777] (IPA).
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
| demo;ecg | -367.63 |
| ecg1;ecg2 | -552.61 |
| demo;annot | -248.52 |

All pairs negative: no synergy anywhere, as in the paper. The screen therefore
selects concatenation; a gated-fusion network (per-modality tanh encoders,
softmax attention, 200 Adam steps on the partial likelihood, 51 params) was
built only to check the screen's prediction:

| fusion | Uno C (tau=7.5) |
|---|---|
| concatenation (Cox, all features) | 0.7408 |
| gated attention | 0.7358 |
| concat - gated | +0.0049 |

Architectural sophistication buys nothing where there is no synergy to
exploit; the extra parameters only add variance.

## (d) Knowledge-graph tensions

| variable | invariant edge | trend edge |
|---|---|---|
| C_tau | theorem invariant | measured trend_down |
| ecg_effect | measured invariant | measured trend_down |

2 structural tension(s) detected: an edge asserting
invariance alongside an edge asserting a trend on the same variable -- the
rule that motivated the null simulation.

## Representation dimension vs label budget (demo)

Gain in Uno C (tau=5) over demo-only on a fixed eval set (n=3000), by PCA
dimension of the ECG block and fit-set label budget:

| budget | 1 PC | 2 PCs | 4 PCs | 9 PCs |
|---|---|---|---|---|
| 400 | +0.0206 | +0.0210 | +0.0404 | +0.0411 |
| 1500 | +0.0233 | +0.0191 | +0.0443 | +0.0476 |
| 4000 | +0.0229 | +0.0216 | +0.0470 | +0.0510 |

## Gated pipeline (blocked run)

Ledger (a failed gate stops the run rather than degrading it):

- 0_provenance: pass
- 1_synergy_screen: blocked: II(demo;ecg)=-367.63 < 0: views redundant, gated fusion not justified; use concatenation

Run completed: False (blocked at the synergy screen: II < 0, so no fusion network is built -- concatenation selected).
