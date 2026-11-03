# BUKPI — Type-Aware Bottom-Up KPI Discovery

## The paper

"Type-Aware Bottom-Up Discovery Pipeline for Cross-Domain Enterprise KPIs"
(under journal revision, 2026).

Enterprise KPI tables mix numeric metrics, categorical dimensions, and free text,
so a single correlation statistic cannot discover relationships bottom-up. The
paper's answer is a type-aware facade: inspect each column pair's types and
dispatch to the association measure suited to that pair, letting candidate KPI
links surface without a pre-declared schema.

## Core contribution implemented

An original implementation of the `correlate(x, y)` facade (no paper text reused):

- numeric x numeric -> Pearson r
- numeric x categorical -> ANOVA eta^2 (between-group SS / total SS)
- categorical x categorical -> Cramer's V (chi^2 contingency table via scipy)
- text x label -> 64-d char-trigram hash sketch, Lloyd k-means (`k=3`, numpy),
  then Cramer's V between cluster ids and the label

Run: `PYTHONPATH=src python3 experiments/bukpi_experiment.py` (seed `20261019`,
`n = 500` rows; sentiment-dense review snippets for the text branch).

## Results

Measured numbers (from `results/bukpi/results.md`):

| branch | planted strength | measure | recovered |
|--------|------------------|---------|-----------|
| numeric-numeric | ~0.70 | Pearson r | 0.728 |
| numeric-categorical | ~0.50 | ANOVA eta^2 | 0.561 |
| categorical-categorical | ~0.60 | Cramer's V | 0.713 |
| text vs sentiment | ~0.95 | cluster Cramer's V | 1.000 |

Null-pair robustness (100 independent numeric pairs): mean |r| = **0.0344**,
max |r| = **0.1006** (threshold 0.2) — PASS.

## Insights

- Every dispatch branch recovered its planted association within 0.15, confirming
  the facade routes each type pair to a statistic that actually measures it.
- The text branch is the most sensitive to data design: with sentiment-dense
  review snippets, 64-d hash sketches separate cleanly (V = 1.000), but with
  diffuse prose sharing most trigrams across classes, hash collisions wash out
  the signal (V ~ 0.12 in an early trial). Sketch dimensionality and text
  density are the operating constraints of that branch.
- The null-pair test shows the pipeline is specific as well as sensitive: pure
  noise never crosses |r| = 0.2, so discovered links are unlikely to be spurious.
