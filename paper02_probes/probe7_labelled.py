#!/usr/bin/env python3
"""Probe 7 -- labelled two-regime probe (manuscript Appendix A,
"A simulation, offered as an illustration": the "Flags are not
discoveries" paragraph).

One run of 200,000 steps per agent on the manuscript's testbed
(implementation B: module ghosh_research.closure of the research
repository), with the ground-truth regime label of every step
recorded, so flag counts split into genuinely novel flags (novel
regime) and known-regime false alarms. The draw sequence is exactly
implementation B's run_agent: a fresh numpy Generator per agent;
z = rng.normal(0, 1, n); if p > 0, mask = rng.random(n) < p and the
masked entries are replaced by rng.normal(4, 1, mask.sum()); a step
flags iff |z| > z_crit with z_crit = Phi^{-1}(1 - alpha/2).
The pure agent is the same code at p = 0 (regime 0 only, so its
novel count is zero by construction of the labels). Seed 12345 for
both agents; deterministic.

Provenance (disclosed): the manuscript's original labelled-probe
artifact (the D-40 build probes, 200,000 steps, seed 12345) no
longer exists in the author's workspace, and its exact counts were
not reproducible from any deposited artifact. Per the elevation
protocol, the figures were therefore re-derived from the deposited
implementation B in this probe (elevation cycle E6, 2026-11-05),
and the manuscript reports this probe's counts verbatim.

Requires Python 3, NumPy and SciPy.
"""

import numpy as np
from scipy.stats import norm

N_STEPS = 200_000
SEED = 12345
P_NOVEL = 0.30
ALPHA = 0.05
NOVEL_MEAN, NOVEL_SD = 4.0, 1.0


def run_labelled(n_steps, p_novel, seed):
    """Return (flags, novel) boolean arrays for one agent run."""
    rng = np.random.default_rng(seed)
    z_crit = float(norm.ppf(1 - ALPHA / 2))
    z = rng.normal(0.0, 1.0, size=n_steps)
    novel = np.zeros(n_steps, dtype=bool)
    if p_novel > 0:
        novel = rng.random(n_steps) < p_novel
        z[novel] = rng.normal(NOVEL_MEAN, NOVEL_SD, size=int(novel.sum()))
    flags = np.abs(z) > z_crit
    return flags, novel


def main():
    z_crit = float(norm.ppf(1 - ALPHA / 2))
    power = float(1 - (norm.cdf(z_crit, NOVEL_MEAN, NOVEL_SD)
                       - norm.cdf(-z_crit, NOVEL_MEAN, NOVEL_SD)))
    print(f"Probe 7 -- labelled two-regime probe (seed {SEED}, "
          f"{N_STEPS} steps per agent)")
    print(f"testbed: known regime N(0,1); novel regime "
          f"N({NOVEL_MEAN:.0f},{NOVEL_SD:.0f}); p = {P_NOVEL}; "
          f"alpha = {ALPHA}; z_crit = {z_crit:.6f}; "
          f"theoretical power beta = {power:.4f}")
    print(f"theory: hybrid rate = p*beta + (1-p)*alpha = "
          f"{P_NOVEL * power + (1 - P_NOVEL) * ALPHA:.4f}; "
          f"precision = p*beta / (p*beta + (1-p)*alpha) = "
          f"{P_NOVEL * power / (P_NOVEL * power + (1 - P_NOVEL) * ALPHA):.4f}")

    flags, novel = run_labelled(N_STEPS, P_NOVEL, SEED)
    n_flags = int(flags.sum())
    n_novel_flags = int((flags & novel).sum())
    n_false = n_flags - n_novel_flags
    print(f"hybrid: steps {N_STEPS}; novel-regime steps "
          f"{int(novel.sum())}; flags {n_flags} "
          f"(rate {n_flags / N_STEPS:.4f}); genuinely novel flags "
          f"{n_novel_flags}; false alarms {n_false}; "
          f"precision {n_novel_flags / n_flags:.4f}; "
          f"recall {n_novel_flags / int(novel.sum()):.4f}")

    flags_p, _ = run_labelled(N_STEPS, 0.0, SEED)
    n_flags_p = int(flags_p.sum())
    print(f"pure: steps {N_STEPS}; flags {n_flags_p} "
          f"(rate {n_flags_p / N_STEPS:.4f}); genuinely novel flags 0")


if __name__ == "__main__":
    main()
