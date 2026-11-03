"""Semantic Chain-of-Thought CAIS: content-aware document region matching.

Original implementation of the core computational idea in
"Semantic Chain-of-Thought CAIS: Vision-Language Document Layout Analysis
without Geometric Ground Truth" (paper under journal revision, 2026).
"""

from .cais import (
    SEGMENT_TYPES,
    make_document_corpus,
    tfidf_matrix,
    cosine_similarity,
    greedy_assignment,
    hungarian_assignment,
    match_quality,
)

__all__ = [
    "SEGMENT_TYPES",
    "make_document_corpus",
    "tfidf_matrix",
    "cosine_similarity",
    "greedy_assignment",
    "hungarian_assignment",
    "match_quality",
]
