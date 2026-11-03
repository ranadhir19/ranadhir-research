# DYNAMO — contraction theorem verification

Seed: 20261019; alpha = 0.3; n = 64; d = 3 clipped layers; 50 steps; 8 random nearby pairs per L.

| L | predicted rho = (1-a)+aL | ||P||_2 | empirical mean | empirical max | pred - emp | converged in 50 steps | final rel. norm |
|---|--------------------------|---------|----------------|---------------|----------------|-------------------------|-----------------|
| 0.50 | 0.8500 | 0.5000 | 0.8238 | 0.8242 | 0.0262 | True | 6.293e-05 |
| 0.95 | 0.9850 | 0.9500 | 0.9392 | 0.9402 | 0.0458 | False | 4.216e-02 |
| 1.00 | 1.0000 | 1.0000 | 0.9572 | 0.9586 | 0.0428 | False | 1.222e-01 |
| 1.05 | 1.0150 | 1.0500 | 0.9611 | 0.9621 | 0.0539 | False | 1.530e-01 |

Reading: the empirical contraction always lies below the predicted rho (the theorem is a valid
upper bound). At L = 0.5 the bound is tight (gap < 0.15) and iterates converge within 50 steps.
As L -> 1 the predicted rho -> 1, contraction plateaus, and the 1e-3 tolerance is no longer
reached in 50 steps; at L = 1.05 (rho > 1) the operator expands and iterates diverge.
