# PINN experiment results

Config: 400 positives, 400 easy negatives, 400 hard negatives, seed=0;
numpy logistic regression, stratified 5-fold CV, pooled out-of-fold AUC.

| protocol | AUC |
|---|---|
| P1: positives vs easy negatives | 1.0000 |
| P2: positives vs balanced easy+hard | 0.9534 |
| P3: positives vs hard negatives | 0.9061 |

Headline accuracy (P1) collapses under the hard-negative stress test (P3).

## Flory-Huggins physics-informed loss demo

Loss spec L = BCE + lam1*|dG_FH/dphi|^2 + lam2*(p_perc - p_c)^2 evaluated on
50 sample composition points (chi=2.5, lam1=lam2=0.1, p_perc=0.42, p_c=0.30):

| term | value |
|---|---|
| BCE | 0.0785 |
| physics gradient term | 0.0042 |
| physics percolation term | 0.0014 |
| total | 0.0842 |

Note: this demonstrates the loss is computable; it is not a trained network.
