# Physics-Informed Neural Networks for LLPS Proteins (pinn)

Original implementation of the core computational ideas from
**"Physics-Informed Neural Networks for Predicting Liquid-Liquid Phase
Separation in Neurodegenerative Disease Proteins"** (paper under journal
revision, 2026). No text or code is taken from the paper; this is an
independent, runnable rendering of its evaluation protocol and its physics
loss specification.

## The paper

The paper tackles predicting which proteins undergo liquid-liquid phase
separation (LLPS) -- a process implicated in neurodegenerative disease --
by coupling a neural classifier to a physics-informed loss derived from
Flory-Huggins polymer theory. Its cautionary finding is methodological:
classifiers that look near-perfect against easy negatives degrade sharply
once evaluated against *hard* negatives that overlap the positive region in
feature space, so reported accuracy depends critically on how the negative
set is constructed.

## Core contribution implemented

Two components:

1. **Hard-negative stress test.** Synthetic protein panels described by four
   biophysical features (disorder content, charge-patterning kappa, aromatic
   fraction, hydropathy). A numpy logistic-regression classifier (L-BFGS via
   scipy, stratified 5-fold CV, pooled out-of-fold AUC) is evaluated under
   three protocols: P1 positives vs easy negatives, P2 positives vs a
   balanced easy+hard mix, P3 positives vs hard negatives only.
2. **Flory-Huggins physics-informed loss (demo).** The paper's loss spec,
   `L = BCE + lam1 * |dG_FH/dphi|^2 + lam2 * (p_perc - p_c)^2` with
   `G_FH(phi) = phi log phi + (1-phi) log(1-phi) + chi phi (1-phi)`,
   evaluated on sample composition points to show every term is computable.
   This is an honest loss-function demonstration, not a trained network.

## How to run

```bash
cd ~/workspace/research
PYTHONPATH=src python3 experiments/pinn_experiment.py   # full experiment
PYTHONPATH=src python3 examples/04_pinn_quickstart.py   # 30-second demo
```

The experiment writes `results/pinn/results.md` and `results/pinn/charts.png`.

## Results

Measured (400 positives / 400 easy / 400 hard negatives, seed 0, 5-fold CV):

| protocol | AUC |
|---|---|
| P1: positives vs easy negatives | 1.0000 |
| P2: positives vs balanced easy+hard | 0.9534 |
| P3: positives vs hard negatives | 0.9061 |

Flory-Huggins loss demo (50 composition points, chi=2.5, lam1=lam2=0.1,
p_perc=0.42, p_c=0.30): BCE = 0.0785, physics gradient term = 0.0042,
physics percolation term = 0.0014, total = 0.0842.

## Insights

- The headline P1 number (AUC 1.000) is the trap the paper warns about: it
  measures separability of the *negative sampling scheme*, not the biology.
  Against hard negatives the same classifier drops to 0.906 -- still
  useful, but a different claim.
- Feature overlap, not model capacity, drives the P3 drop: the hard
  negatives were drawn from Gaussians shifted by well under one standard
  deviation per feature, and a linear model is the right tool to show that
  the difficulty is in the data geometry.
- The physics loss terms are small but nonzero and smooth in phi, which is
  precisely their role: a gentle regularizer pulling predictions toward
  thermodynamically consistent phase behavior rather than a dominant
  objective that would overwhelm the data term.
