"""Hybrid-discovery simulation (Agent Closure Theorem).

Original implementation of the computational core of
"The Agent Closure Theorem: Mathematical Bounds on AI Knowledge Generation
and a Hybrid-Discovery Pathway" (paper under journal revision, 2026).

The paper is purely theoretical: it proves that a pure-AI discovery agent,
whose observations are confined to its training prior, cannot generate
knowledge outside that prior's support at a rate above the significance
level alpha of any novelty test (the "closure" bound), while a hybrid
agent that occasionally observes a genuinely novel regime can. This module
implements the paper's canonical two-regime simulation as the closest
faithful computational demonstration:

- regime 0 ("known"):   observations z ~ N(0, 1)
- regime 1 ("novel"):   observations z ~ N(4, 1)

A pure-AI agent samples only regime 0. A hybrid agent observes regime 1 with
probability p_novel per step. Both agents apply the same novelty test --
reject the N(0, 1) prior when |z| > z_{1-alpha/2} -- and we measure the
fraction of steps flagged novel. Theory predicts pure ~= alpha and
hybrid ~= p_novel * P(|N(4,1)| > z_crit) + (1 - p_novel) * alpha.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = [
    "novelty_threshold",
    "theoretical_novel_rates",
    "run_agent",
    "simulate",
]

Z_CRIT_DEFAULT = float(norm.ppf(1 - 0.05 / 2))  # 1.9599...


def novelty_threshold(alpha: float = 0.05) -> float:
    """Two-sided critical value |z| > z_crit under the N(0,1) prior."""
    return float(norm.ppf(1 - alpha / 2))


def theoretical_novel_rates(
    p_novel: float = 0.30, alpha: float = 0.05,
    novel_mean: float = 4.0, novel_sd: float = 1.0,
) -> dict:
    """Closed-form expected novel-flag rates for both agents.

    Pure agent: exactly alpha (it can only false-positive under its prior).
    Hybrid: mixture of the detection power on the novel regime and alpha on
    the known regime.
    """
    z_crit = novelty_threshold(alpha)
    power = float(1 - (norm.cdf(z_crit, novel_mean, novel_sd)
                       - norm.cdf(-z_crit, novel_mean, novel_sd)))
    return {
        "pure": alpha,
        "hybrid": p_novel * power + (1 - p_novel) * alpha,
        "detection_power": power,
        "z_crit": z_crit,
    }


def run_agent(
    n_steps: int,
    p_novel: float,
    rng: np.random.Generator,
    alpha: float = 0.05,
    novel_mean: float = 4.0,
    novel_sd: float = 1.0,
) -> float:
    """Simulate one agent for n_steps; return its novel-flag rate.

    ``p_novel = 0`` gives the pure-AI agent (regime 0 only); ``p_novel > 0``
    gives the hybrid agent, which observes regime 1 with probability
    ``p_novel`` per step.
    """
    z_crit = novelty_threshold(alpha)
    z = rng.normal(0.0, 1.0, size=n_steps)
    if p_novel > 0:
        novel_mask = rng.random(n_steps) < p_novel
        z[novel_mask] = rng.normal(novel_mean, novel_sd, size=int(novel_mask.sum()))
    return float(np.mean(np.abs(z) > z_crit))


def simulate(
    n_steps: int = 2000,
    n_seeds: int = 30,
    p_novel: float = 0.30,
    alpha: float = 0.05,
    base_seed: int = 0,
) -> dict:
    """Run both agents over ``n_seeds`` independent seeds.

    Returns per-seed rates plus mean and 95% normal-approximation confidence
    intervals for each agent.
    """
    pure_rates = np.empty(n_seeds)
    hybrid_rates = np.empty(n_seeds)
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        pure_rates[s] = run_agent(n_steps, 0.0, rng, alpha)
        rng = np.random.default_rng(base_seed + 10_000 + s)
        hybrid_rates[s] = run_agent(n_steps, p_novel, rng, alpha)

    def stats(rates: np.ndarray) -> dict:
        mean = float(rates.mean())
        se = float(rates.std(ddof=1) / np.sqrt(n_seeds))
        return {"mean": mean, "ci95": 1.96 * se, "seeds": rates}

    return {
        "pure": stats(pure_rates),
        "hybrid": stats(hybrid_rates),
        "n_steps": n_steps,
        "n_seeds": n_seeds,
        "p_novel": p_novel,
        "alpha": alpha,
        "theory": theoretical_novel_rates(p_novel, alpha),
    }
