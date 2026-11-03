# Theoretical Augmentation for Robust Option Pricing (dsai)

Original implementation of the core computational idea from
**"Theoretical Augmentation Bridging Analytic Finance and Artificial
Intelligence for Robust Option Pricing in Extreme Market Regimes"**
(paper under journal revision, 2026). No text or code is taken from the
paper; this is an independent, runnable rendering of its hybrid pricing
architecture and regime-shift evaluation.

## The paper

Option pricing models trained on calm markets tend to fail when volatility
regimes shift, because a pure machine-learning pricer has no structural
anchor -- it extrapolates a fitted surface into regions it has never seen.
The paper's proposal is theoretical augmentation: price as
`C_hat = C_BS + f_theta(x)`, where the analytic Black-Scholes formula
carries the dominant, well-understood structure and a data-driven model
learns only the *residual* (smile, skew, liquidity effects). Under regime
shift, the theory anchor holds while the residual correction degrades
gracefully, instead of the whole price collapsing.

## Core contribution implemented

`C_hat = C_BS + f_theta(x)` with a numpy Black-Scholes call pricer and a
ridge-regression residual model on (moneyness, maturity, rate, vol, VIX):

- Synthetic panel, n=4000 options per regime. Calm: price = C_BS + mild
  skew + small noise (VIX 12-22). Crisis: price = C_BS + amplified
  volatility-smile skew + shock (VIX 30-60).
- Train **only** on calm; test on crisis. Compare (a) pure-ML ridge on
  price, (b) BS-only, (c) hybrid BS + ridge residual. Metrics: RMSE / MAE.

## How to run

```bash
cd ~/workspace/research
PYTHONPATH=src python3 experiments/dsai_experiment.py   # full experiment
PYTHONPATH=src python3 examples/05_dsai_quickstart.py   # 30-second demo
```

The experiment writes `results/dsai/results.md` and `results/dsai/charts.png`.

## Results

Measured (train calm n=4000, test crisis n=4000, seed 0, ridge lambda=1.0):

| model | crisis RMSE | crisis MAE |
|---|---|---|
| pure ML (ridge on price) | 4.730 | 3.792 |
| BS-only (analytic theory) | 3.825 | 3.498 |
| hybrid (BS + ridge residual) | 3.652 | 3.325 |

Training (calm) RMSE: pure ML 2.752, hybrid 0.251.

## Insights

- The pure-ML model looks competitive in-sample (calm RMSE 2.75) and then
  loses ~72% of that accuracy out-of-regime -- the classic "fits the calm
  surface, not the pricing function" failure. Its linear extrapolation in
  VIX cannot reproduce the crisis smile.
- The hybrid's edge over BS-only is modest in absolute terms (RMSE 3.65 vs
  3.83) but conceptually decisive: the residual learner picked up the
  *shape* of the skew from calm data (where it appears at 0.2 amplitude)
  and extrapolated that shape -- not the price level -- into the crisis.
- Decomposing the problem this way also makes the model auditable: the
  theory component is exact and inspectable, and the ML component's
  contribution is bounded and attributable to named features, which matters
  for model-risk governance in a way a monolithic pricer cannot match.
