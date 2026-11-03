"""Content-aware region matching (Semantic Chain-of-Thought CAIS).

Original implementation of the core computational idea in
"Semantic Chain-of-Thought CAIS: Vision-Language Document Layout Analysis
without Geometric Ground Truth" (paper under journal revision, 2026).

The paper performs document layout analysis without bounding-box ground
truth by matching *predicted* regions to *reference* segments on the basis
of semantic content rather than geometry. This module renders that idea as
a reproducible simulation:

1. Synthetic "documents": N reference segments, each carrying a natural
   language description drawn from a financial-document vocabulary
   (tables, flowcharts, code blocks, paragraphs, charts, ...).
2. K predicted regions with noisy descriptions (word dropout / word swaps,
   plus spurious regions with random content), mimicking a vision-language
   model's region proposals.
3. TF-IDF vectors for all descriptions, a K x N cosine-similarity matrix,
   and optimal one-to-one assignment via the Hungarian algorithm
   (``scipy.optimize.linear_sum_assignment``), compared against a greedy
   per-region argmax baseline.

Metric: mean cosine similarity of the matched pairs; the experiment ablates
Hungarian matching on/off (greedy baseline vs optimal assignment).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = [
    "SEGMENT_TYPES",
    "make_document_corpus",
    "tfidf_matrix",
    "cosine_similarity",
    "greedy_assignment",
    "hungarian_assignment",
    "match_quality",
]

# Financial-document region vocabulary: segment type -> description templates
# with {slot} fillers drawn from the pools below.
SEGMENT_TYPES: dict[str, list[str]] = {
    "table": [
        "revenue table showing {metric} by {period} across {region}",
        "balance sheet table with {metric} totals for {period}",
    ],
    "flowchart": [
        "flowchart of the {process} workflow with decision diamonds",
        "process flowchart illustrating {process} approval steps",
    ],
    "code": [
        "python code snippet for {process} data extraction",
        "sql code block querying {metric} from the warehouse",
    ],
    "paragraph": [
        "paragraph describing {metric} risk factors in {region}",
        "text paragraph summarizing {period} {metric} performance",
    ],
    "chart": [
        "bar chart of {metric} volume per {period} in {region}",
        "line chart tracking {metric} trends over {period}",
    ],
    "form": [
        "application form for {process} with signature fields",
        "compliance form capturing {metric} disclosures",
    ],
    "invoice": [
        "invoice with line items for {metric} billed in {period}",
        "payment invoice summarizing {metric} charges by {region}",
    ],
    "diagram": [
        "network diagram of {process} service dependencies",
        "architecture diagram showing {metric} data flow",
    ],
}

_FILLERS: dict[str, list[str]] = {
    "metric": ["quarterly earnings", "loan exposure", "transaction volume",
               "credit losses", "fee income", "operating margin"],
    "period": ["fiscal quarter", "calendar month", "reporting year",
               "trailing twelve months"],
    "region": ["north america", "europe", "asia pacific", "latin america"],
    "process": ["loan approval", "fraud review", "kyc onboarding",
                "settlement reconciliation"],
}

_VOCAB_POOL = sorted({w for pool in _FILLERS.values() for phrase in pool
                      for w in phrase.split()}
                     | {"table", "chart", "form", "code", "text", "figure",
                        "summary", "report", "audit", "ledger", "policy"})


def _render_template(template: str, rng: np.random.Generator) -> str:
    out = template
    for slot, pool in _FILLERS.items():
        out = out.replace("{" + slot + "}", rng.choice(pool))
    return out


def _noise_description(
    desc: str, rng: np.random.Generator,
    dropout: float = 0.12, swap: float = 0.05,
) -> str:
    """Simulate a noisy VLM region description: word dropout and swaps."""
    words = desc.split()
    kept = [w for w in words if rng.random() > dropout]
    if not kept:
        kept = [rng.choice(words)]
    noisy = [rng.choice(_VOCAB_POOL) if rng.random() < swap else w for w in kept]
    return " ".join(noisy)


def make_document_corpus(
    n_segments: int = 40,
    n_spurious: int = 5,
    dropout: float = 0.12,
    swap: float = 0.05,
    seed: int = 0,
) -> dict:
    """Build synthetic reference segments and noisy predicted regions.

    Returns ``gt_texts`` (N reference descriptions), ``pred_texts`` (K
    predicted descriptions), and ``pred_source`` (the ground-truth index
    each prediction was derived from, or -1 for spurious regions).
    """
    rng = np.random.default_rng(seed)
    types = sorted(SEGMENT_TYPES)
    gt_texts = []
    for i in range(n_segments):
        seg_type = types[i % len(types)]
        template = rng.choice(SEGMENT_TYPES[seg_type])
        gt_texts.append(f"{seg_type} " + _render_template(template, rng))

    pred_texts, pred_source = [], []
    for j, gt in enumerate(gt_texts):
        pred_texts.append(_noise_description(gt, rng, dropout, swap))
        pred_source.append(j)
    for _ in range(n_spurious):
        n_words = int(rng.integers(8, 15))
        pred_texts.append(" ".join(rng.choice(_VOCAB_POOL, size=n_words)))
        pred_source.append(-1)
    return {
        "gt_texts": gt_texts,
        "pred_texts": pred_texts,
        "pred_source": np.array(pred_source),
    }


def tfidf_matrix(texts: list[str]) -> tuple[np.ndarray, list[str]]:
    """Manual TF-IDF: raw term counts x smoothed idf, rows L2-normalized."""
    tokenized = [t.lower().split() for t in texts]
    vocab = sorted({tok for doc in tokenized for tok in doc})
    index = {tok: i for i, tok in enumerate(vocab)}
    D = len(texts)
    tf = np.zeros((D, len(vocab)))
    for d, doc in enumerate(tokenized):
        for tok in doc:
            tf[d, index[tok]] += 1.0
    df = (tf > 0).sum(axis=0)
    idf = np.log((1 + D) / (1 + df)) + 1.0
    mat = tf * idf
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms, vocab


def cosine_similarity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between L2-normalized row sets A and B."""
    return A @ B.T


def greedy_assignment(sim: np.ndarray) -> list[tuple[int, int, float]]:
    """Baseline: each predicted region takes its argmax reference segment.

    Allows collisions (several predictions may claim the same segment).
    Returns (pred_idx, gt_idx, similarity) triples.
    """
    best = np.argmax(sim, axis=1)
    return [(i, int(best[i]), float(sim[i, best[i]])) for i in range(sim.shape[0])]


def hungarian_assignment(sim: np.ndarray) -> list[tuple[int, int, float]]:
    """Optimal one-to-one assignment maximizing total similarity.

    Uses ``scipy.optimize.linear_sum_assignment`` (the Hungarian algorithm)
    on the negated similarity (cost) matrix; handles K != N.
    """
    rows, cols = linear_sum_assignment(-sim)
    return [(int(r), int(c), float(sim[r, c])) for r, c in zip(rows, cols)]


def match_quality(pairs: list[tuple[int, int, float]]) -> dict:
    """Mean matched-pair similarity and collision statistics."""
    sims = np.array([p[2] for p in pairs])
    gt_ids = [p[1] for p in pairs]
    n_collisions = len(gt_ids) - len(set(gt_ids))
    return {
        "mean_similarity": float(sims.mean()) if len(sims) else 0.0,
        "n_pairs": len(pairs),
        "n_collisions": n_collisions,
        "collision_rate": float(n_collisions / len(pairs)) if pairs else 0.0,
    }
