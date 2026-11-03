# DSAI experiment results

Config: train on calm regime (n=4000), test on crisis regime (n=4000),
seed=0; ridge lambda=1.0 on (moneyness, maturity, rate, vol, VIX).

| model | crisis RMSE | crisis MAE |
|---|---|---|
| pure ML (ridge on price) | 4.730 | 3.792 |
| BS-only (analytic theory) | 3.825 | 3.498 |
| hybrid (BS + ridge residual) | 3.652 | 3.325 |

Training (calm) RMSE: pure ML 2.752, hybrid
0.251. The pure-ML pricer fits calm data well but
collapses under the crisis regime shift; the theory anchor keeps the hybrid
model robust and best on crisis data.
