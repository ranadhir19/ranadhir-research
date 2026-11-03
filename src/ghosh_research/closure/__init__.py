"""Agent Closure Theorem: hybrid-discovery simulation.

Original implementation of the computational core of
"The Agent Closure Theorem: Mathematical Bounds on AI Knowledge Generation
and a Hybrid-Discovery Pathway" (paper under journal revision, 2026).

Note: the paper itself is purely theoretical; this module implements its
canonical two-regime simulation as the closest faithful computational
demonstration of the closure bound.
"""

from .closure import (
    novelty_threshold,
    theoretical_novel_rates,
    run_agent,
    simulate,
)

__all__ = [
    "novelty_threshold",
    "theoretical_novel_rates",
    "run_agent",
    "simulate",
]
