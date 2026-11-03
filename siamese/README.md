# SIAMESE — Contrastive Verification for Bank-Cheque CAR/LAR Amounts

## The paper

"Siamese Networks for Bank-Cheque CAR/LAR Verification" (under journal revision,
2026).

A cheque states its amount twice: the CAR ("courtesy amount recognition",
`$125.00`) and the LAR ("legal amount recognition", "one hundred twenty five
dollars"). Verification means deciding whether the two readings encode the same
amount. The paper frames this as contrastive pair verification and asks how far
a *linear* similarity model can go before deep encoders are needed.

## Core contribution implemented

An original implementation of the paper's verification pipeline (no paper text
reused), all in numpy:

- Synthetic pair generator: amounts -> CAR/LAR strings via an English
  number-to-words routine; negatives are same-magnitude mismatched amounts
  (hard negatives, removing the trivial amount-length shortcut).
- Bigram-hash character features (256-d, signed); each pair is represented by
  `|f_car - f_lar|`.
- 16-d linear projection trained with contrastive loss (margin 1.0) by SGD.
- Two honest baselines: raw `|f_car - f_lar|` distance (threshold tuned on
  train) and a purely linear classifier (logistic regression) on the pair vectors.

Run: `PYTHONPATH=src python3 experiments/siamese_experiment.py` (seeds
`20261019`-`20261023`; 4000 train / 1500 test pairs per seed). Takes about 40 seconds.

## Results

Measured numbers (from `results/siamese/results.md`, mean +/- std over 5 seeds):

| method | test accuracy |
|--------|---------------|
| raw-distance baseline | 0.5027 +/- 0.0080 |
| linear classifier | 0.5075 +/- 0.0123 |
| siamese embedding (contrastive) | 0.9800 +/- 0.0021 |

## Insights

- **Deviation from the brief's expectation, reported as measured.** The brief
  (following the paper) expected the linear embedding to sit near chance
  (~54%). It does not: it reaches 0.98 on clean synthetic pairs, consistently
  across all 5 seeds, and 0.95 even when trained and tested on disjoint amount
  ranges.
- The mechanism is identifiable: `||W d||^2 = d' W W' d` makes the "linear
  projection + norm" siamese a *rank-16 quadratic* model. The contrastive
  objective uses that capacity to learn compositional digit<->word bigram
  correspondences (e.g. digit-bigram "25" co-occurring with word-bigrams from
  "twenty"/"five"), which compose across unseen amounts.
- The paper's "linear similarity is insufficient" intuition *does* hold for
  genuinely linear readouts on this task: the raw-distance baseline (0.503) and
  the linear classifier (0.508) sit at chance, since neither can express the
  AND of "these digit-bigrams AND those word-bigrams co-occur".
- The honest reading: on noiseless synthetic pairs the correspondence signal is
  clean enough for a quadratic model to crack. The paper's ~54% presumably
  reflects real cheque noise — OCR errors, handwriting variation, abbreviations
  — which this benchmark does not model. That gap, not the encoder depth, is
  where the paper's deep-encoder motivation would need to be re-tested.
