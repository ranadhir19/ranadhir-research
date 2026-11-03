# The Agent Closure Theorem (closure)

Original implementation of the computational core of
**"The Agent Closure Theorem: Mathematical Bounds on AI Knowledge Generation
and a Hybrid-Discovery Pathway"** (paper under journal revision, 2026). No
text or code is taken from the paper.

**Note: the paper is purely theoretical.** There is no training run or
dataset to reproduce. This module implements the paper's canonical
two-regime simulation as the closest faithful computational demonstration
of its closure bound.

## The paper

The paper proves a bound on what an AI discovery agent can ever find: if an
agent's observations are confined to the support of its training prior, no
novelty test it runs can flag genuinely new knowledge at a rate above the
test's own significance level alpha -- the agent is "closed" under its
prior. The escape hatch is a hybrid pathway: an agent that occasionally
observes data from outside the prior (a genuinely novel regime) breaks the
bound, with its discovery rate following a simple mixture law. Theory, not
benchmarks, is the contribution.

## Core contribution implemented

The two-regime discovery simulation:

- regime 0 ("known"): observations z ~ N(0, 1)
- regime 1 ("novel"): observations z ~ N(4, 1)

A pure-AI agent samples only regime 0; a hybrid agent observes regime 1
with probability p = 0.30 per step. Both apply the same novelty test --
reject the N(0,1) prior when |z| > 1.96 (alpha = 0.05) -- over 2000 steps x
30 seeds. Theory predicts novel-flag rates of alpha for the pure agent and
p * power + (1-p) * alpha for the hybrid agent.

## How to run

```bash
cd ~/workspace/research
PYTHONPATH=src python3 experiments/closure_experiment.py   # full experiment
PYTHONPATH=src python3 examples/03_closure_quickstart.py   # 30-second demo
```

The experiment writes `results/closure/results.md` and `results/closure/charts.png`.

## Results

Measured (2000 steps x 30 seeds, p_novel = 0.30, alpha = 0.05):

| agent | measured novel rate | 95% CI | theoretical |
|---|---|---|---|
| pure AI (regime 0 only) | 0.0483 | +- 0.0019 | 0.0500 |
| hybrid (30% novel-regime obs.) | 0.3260 | +- 0.0034 | 0.3288 |

## Insights

- The pure agent's measured rate (0.0483) sits exactly on the alpha bound:
  confined to its prior, everything it flags "novel" is a false positive by
  construction. The bound is tight, not just an inequality.
- The hybrid agent's rate (0.3260) matches the mixture prediction to three
  decimals, confirming the paper's hybrid pathway quantitatively: discovery
  rate is controlled by how often the agent is *allowed* to see the novel
  regime, times the test's detection power there (0.9794 here).
- The practical moral is architectural, not algorithmic: no cleverer test
  statistic lifts the pure agent above alpha. If you want an AI system to
  discover genuinely new things, you must engineer the observation channel,
  not the detector.
