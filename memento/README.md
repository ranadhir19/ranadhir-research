# MEMENTO — Continual Learning: Retention vs Plasticity

## The paper

"Multi-Agent Memento" (under journal revision, 2026).

Continual learning pits two goals against each other: retain what was learned
early (no catastrophic forgetting) and stay plastic enough to track
distribution shift. The paper's honest empirical finding is that these goals
trade off — neither a fine-tuning analogue nor a never-forget memory store
dominates everywhere.

## Core contribution implemented

An original implementation of the paper's comparison (no paper text reused), on
a synthetic binary stream (`n = 2000`, features uniform on `[-1, 1]^2`) whose
class boundary rotates 45 -> 90 -> 135 degrees at `t = 700` and `t = 1400`
(5% label noise):

- (a) **Sliding-window refit** (fine-tune analogue): logistic regression by
  mini-batch SGD in numpy, refit every 200 steps on the last 400 samples,
  warm-started from the previous weights.
- (b) **Prototype bank** (no-forgetting analogue): every 200-sample block stores
  its per-class feature means; nothing is ever deleted. Prediction is a
  similarity-weighted vote, `argmax_c x . M_c`.

Run: `PYTHONPATH=src python3 experiments/memento_experiment.py` (seed `20261019`).

## Results

Measured numbers (from `results/memento/results.md`):

| method | overall | 0-700 | 700-1400 | 1400-2000 |
|--------|---------|-------|----------|-----------|
| sliding-window refit | 0.567 | 0.526 | 0.299 | 0.927 |
| prototype bank | 0.715 | 0.914 | 0.661 | 0.543 |

## Insights

- Stated explicitly, as the paper does: **this is a trade-off, not a clean win.**
  The prototype bank retains early knowledge (0.914 / 0.661 on the first two
  segments, long after the shifts) but its frozen compromise — stale prototypes
  keep voting — lags badly on the latest distribution (0.543). The refit adapts
  to the current segment (0.927) while forgetting the earlier ones (0.526, and
  0.299 on the middle segment, where the new boundary's orientation is nearly
  reversed relative to the old one).
- Which method is "better" depends on the deployment question: all data ever
  seen (memory wins, 0.715 vs 0.567) or the distribution right now (refit wins,
  0.927 vs 0.543). Any continual-learning claim has to name which one it is
  optimizing.
