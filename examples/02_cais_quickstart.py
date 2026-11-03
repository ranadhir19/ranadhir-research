#!/usr/bin/env python3
"""Quickstart: content-aware region matching (Semantic CoT-CAIS).

Builds a small synthetic document corpus, matches noisy predicted regions
to reference segments with greedy argmax vs Hungarian assignment, and prints
the ablation. Runs in well under 60 seconds.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ghosh_research import cais

corpus = cais.make_document_corpus(n_segments=16, n_spurious=3, seed=7)
texts = corpus["gt_texts"] + corpus["pred_texts"]
mat, vocab = cais.tfidf_matrix(texts)
sim = cais.cosine_similarity(mat[16:], mat[:16])

q_greedy = cais.match_quality(cais.greedy_assignment(sim))
q_hung = cais.match_quality(cais.hungarian_assignment(sim))

print(f"greedy argmax:      mean similarity = {q_greedy['mean_similarity']:.4f}, "
      f"collisions = {q_greedy['n_collisions']}")
print(f"Hungarian:          mean similarity = {q_hung['mean_similarity']:.4f}, "
      f"collisions = {q_hung['n_collisions']}")
print(f"Hungarian gain: {q_hung['mean_similarity'] - q_greedy['mean_similarity']:+.4f}")
