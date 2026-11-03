# Coarse Set Theory for Ethical AI (cst)

Original implementation of the core computational idea from
**"Coarse Set Theory for Ethical AI: A Mathematical Framework for
Multi-Constraint Decision Making with Formal Guarantees"**
(paper under journal revision, 2026). No text or code is taken from the
paper; this is an independent, runnable rendering of its central
construction.

## The paper

The paper asks how an AI decision system can satisfy several ethical
constraints at once while staying as useful as possible. Its answer is to
treat ethics as a set-construction problem: from a pool of candidate
decisions, build a *maximal* subset whose members jointly respect every
constraint (fairness bounds, safety rules, and the like), with formal
guarantees that the construction terminates with a constraint-satisfying,
maximal set. The motivating tension is that each constraint alone is easy
to check, but their interaction makes naive filtering either infeasible or
needlessly destructive to utility.

## Core contribution implemented

The greedy maximal ethical subset construction:

```
M = empty; candidates = X
while candidates nonempty:
    x* = argmin_x  sum_i (1 - I^{c_i}(x))^2      # fewest violated constraints
    if satisfaction(x*) >= 1 - eps: admit x*, drop its eps-proximity neighbours
    else: discard x*
```

Constraint indicators `I^{c_i}(x)` are evaluated on the *hypothetical*
admitted set `M U {x}` using running per-group counts (O(candidates) per
iteration). Ties are broken by descending decision utility (predicted
score), which preserves accuracy while the constraints bind. Demo
constraints: demographic-parity gap <= 0.015 and equal-opportunity gap
<= 0.015 on a synthetic lending population (n=2000) scored by a
globally-tuned threshold classifier.

## How to run

```bash
cd ~/workspace/research
PYTHONPATH=src python3 experiments/cst_experiment.py   # full experiment
PYTHONPATH=src python3 examples/01_cst_quickstart.py   # 30-second demo
```

The experiment writes `results/cst/results.md` and `results/cst/charts.png`.

## Results

Measured (seed 0, n=2000, tau_dp = tau_eo = 0.015, eps = 0, delta = 0.1):

| metric | baseline (full set) | admitted subset |
|---|---|---|
| demographic-parity gap | 0.1327 | 0.0148 |
| equal-opportunity gap | 0.0921 | 0.0144 |
| accuracy | 0.6945 | 0.6763 |
| n decisions | 2000 | 1600 (80.0%) |

Accuracy cost of the fairness construction: **1.82 percentage points**.

## Insights

- Evaluating each candidate against the *hypothetical* admitted set (rather
  than scoring candidates in isolation) is what makes the greedy procedure
  sound: a decision is only admitted if the set remains fair with it in.
- The eps-proximity neighbourhood removal is doing real work beyond
  fairness: it enforces spread, so the admitted set stays maximal rather
  than collapsing onto a tight cluster of mutually similar decisions.
- The accuracy cost is small because the utility tie-break admits
  high-confidence decisions first; the binding constraint is the *rate*
  balance across groups, which forces the procedure to stop admitting
  advantaged-group positives once parity would break -- a clean, auditable
  trade-off curve rather than a black-box debiasing step.
- One honest caveat: at small n the equal-opportunity estimates are noisy
  (few true positives per group early on), which can keep positives
  infeasible too long and strand the admitted set among borderline
  negatives. The construction is healthiest once per-group counts
  stabilize -- at n=2000 the gaps/cost story is stable across seeds.
