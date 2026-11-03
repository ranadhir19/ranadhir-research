"""bukpi — type-aware bottom-up discovery pipeline for enterprise KPIs.

Original implementation of the core idea of "Type-Aware Bottom-Up
Discovery Pipeline for Cross-Domain Enterprise KPIs" (under journal
revision, 2026): a ``correlate()`` facade that dispatches on column types
to Pearson r, ANOVA eta^2, Cramer's V, or a hash-sketch + k-means text
branch.
"""

from .bukpi import (
    pearson_r,
    anova_eta2,
    cramers_v,
    hash_sketch,
    lloyd_kmeans,
    text_cluster_v,
    correlate,
)

__all__ = [
    "pearson_r",
    "anova_eta2",
    "cramers_v",
    "hash_sketch",
    "lloyd_kmeans",
    "text_cluster_v",
    "correlate",
]
