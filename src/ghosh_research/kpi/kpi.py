"""Bayesian validation of concept-map edges for KPI governance.

A concept-map edge asserts a linear causal link ``y = beta * x + eps``
between two KPIs.  This module validates each edge with a Bayesian linear
regression fitted by a random-walk Metropolis-Hastings sampler over
``(beta, log sigma)`` in numpy, and scores the edge's existence with a
Savage-Dickey Bayes factor

    BF10 ~= N(0; 0, sigma_beta) / N(0; beta_hat, sd(beta_hat)),

the ratio of the prior density at ``beta = 0`` to the (normal-approximated)
marginal posterior density at ``beta = 0``.  Large ``BF01 = 1 / BF10``
means evidence *for* the null (no edge).

This is an original implementation of the paper's core idea, written from
the method description; the paper is under journal revision (2026).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def log_posterior(beta: float, log_sigma: float, x: np.ndarray, y: np.ndarray,
                  prior_beta_sd: float) -> float:
    """Log posterior for ``y = beta * x + eps``, priors ``beta ~ N(0, s_b^2)``,
    ``log sigma ~ N(0, 1)``."""
    sigma = np.exp(log_sigma)
    n = x.size
    resid = y - beta * x
    loglik = -0.5 * n * np.log(2.0 * np.pi) - n * log_sigma \
        - 0.5 * np.sum(resid ** 2) / sigma ** 2
    logprior = norm.logpdf(beta, 0.0, prior_beta_sd) + norm.logpdf(log_sigma, 0.0, 1.0)
    return float(loglik + logprior)


def mh_sample(x: np.ndarray, y: np.ndarray, rng: np.random.Generator,
              prior_beta_sd: float = 10.0, n_iter: int = 25000,
              burn: int = 8000) -> dict:
    """Random-walk Metropolis-Hastings sampler for ``(beta, log sigma)``.

    The chain is initialised at the OLS solution with proposal scales set
    from the OLS standard error (Laplace-informed start).  Returns posterior
    samples and diagnostics.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # OLS start and proposal scales
    beta_ols = float(np.dot(x, y) / np.dot(x, x))
    resid = y - beta_ols * x
    sigma_ols = float(np.sqrt(np.mean(resid ** 2)))
    se_ols = sigma_ols / float(np.sqrt(np.dot(x, x)))
    prop_beta = 2.4 * max(se_ols, 1e-8)
    prop_logsigma = 0.10

    beta, log_sigma = beta_ols, float(np.log(max(sigma_ols, 1e-8)))
    lp = log_posterior(beta, log_sigma, x, y, prior_beta_sd)
    betas = np.empty(n_iter)
    sigmas = np.empty(n_iter)
    accepts = 0
    for i in range(n_iter):
        b_prop = beta + rng.normal(0.0, prop_beta)
        ls_prop = log_sigma + rng.normal(0.0, prop_logsigma)
        lp_prop = log_posterior(b_prop, ls_prop, x, y, prior_beta_sd)
        if np.log(rng.random()) < lp_prop - lp:
            beta, log_sigma, lp = b_prop, ls_prop, lp_prop
            accepts += 1
        betas[i] = beta
        sigmas[i] = np.exp(log_sigma)
    return {
        "beta": betas[burn:],
        "sigma": sigmas[burn:],
        "accept_rate": accepts / n_iter,
    }


def savage_dickey_bf10(beta_samples: np.ndarray, prior_beta_sd: float) -> float:
    """Savage-Dickey BF10 for H1: beta != 0 vs H0: beta = 0.

    Approximates the marginal posterior of ``beta`` by a normal with the
    sample mean and sd, then returns ``p(beta=0 | H1 prior) / p(beta=0 | data)``.
    Computed in log space so overwhelming evidence saturates at ``inf``
    instead of underflowing.
    """
    bhat = float(np.mean(beta_samples))
    bsd = max(float(np.std(beta_samples)), 1e-12)
    log_bf10 = float(norm.logpdf(0.0, 0.0, prior_beta_sd) - norm.logpdf(0.0, bhat, bsd))
    return float(np.exp(log_bf10)) if log_bf10 < 700 else float("inf")


def hdi_central(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Central ``level`` credible interval (equal-tailed) of the samples."""
    lo = (1.0 - level) / 2.0 * 100.0
    hi = (1.0 + level) / 2.0 * 100.0
    return float(np.percentile(samples, lo)), float(np.percentile(samples, hi))


def validate_edge(x: np.ndarray, y: np.ndarray, rng: np.random.Generator,
                  prior_beta_sd: float = 10.0, n_iter: int = 25000,
                  burn: int = 8000) -> dict:
    """Fit one concept-map edge and return its validation summary."""
    post = mh_sample(x, y, rng, prior_beta_sd, n_iter, burn)
    beta = post["beta"]
    lo, hi = hdi_central(beta)
    bf10 = savage_dickey_bf10(beta, prior_beta_sd)
    return {
        "beta_hat": float(np.mean(beta)),
        "beta_sd": float(np.std(beta)),
        "hdi_lo": lo,
        "hdi_hi": hi,
        "bf10": bf10,
        "bf01": 1.0 / bf10,
        "accept_rate": post["accept_rate"],
    }
