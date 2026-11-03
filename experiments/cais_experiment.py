#!/usr/bin/env python3
"""CAIS experiment: content-aware region matching with Hungarian assignment.

Builds synthetic document reference segments and noisy predicted regions,
computes a TF-IDF cosine-similarity matrix, and compares greedy per-region
argmax matching against optimal Hungarian assignment. Ablates Hungarian
matching on/off. Writes results/cais/results.md and results/cais/charts.png.
All randomness is seeded.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ghosh_research import cais  # noqa: E402

SEED = 0
N_SEGMENTS = 40
N_SPURIOUS = 5
OUT = Path(__file__).resolve().parents[1] / "results" / "cais"


def main() -> dict:
    corpus = cais.make_document_corpus(
        n_segments=N_SEGMENTS, n_spurious=N_SPURIOUS, seed=SEED
    )
    texts = corpus["gt_texts"] + corpus["pred_texts"]
    mat, vocab = cais.tfidf_matrix(texts)
    G = mat[:N_SEGMENTS]
    P = mat[N_SEGMENTS:]
    sim = cais.cosine_similarity(P, G)
    print(f"similarity matrix: {sim.shape}, vocab size: {len(vocab)}")

    greedy_pairs = cais.greedy_assignment(sim)
    hungarian_pairs = cais.hungarian_assignment(sim)
    q_greedy = cais.match_quality(greedy_pairs)
    q_hung = cais.match_quality(hungarian_pairs)

    print(f"greedy:    mean matched similarity={q_greedy['mean_similarity']:.4f} "
          f"collision_rate={q_greedy['collision_rate']:.3f}")
    print(f"hungarian: mean matched similarity={q_hung['mean_similarity']:.4f} "
          f"collision_rate={q_hung['collision_rate']:.3f}")
    gain = q_hung["mean_similarity"] - q_greedy["mean_similarity"]
    print(f"Hungarian gain over greedy: {gain:+.4f}")

    OUT.mkdir(parents=True, exist_ok=True)
    md = f"""# CAIS experiment results

Config: {N_SEGMENTS} reference segments, {sim.shape[0]} predicted regions
({N_SPURIOUS} spurious), seed={SEED}, TF-IDF vocabulary size {len(vocab)}.

| matching | mean matched-pair similarity | pairs | collision rate |
|---|---|---|---|
| greedy argmax (Hungarian OFF) | {q_greedy['mean_similarity']:.4f} | {q_greedy['n_pairs']} | {q_greedy['collision_rate']:.3f} |
| Hungarian assignment (ON) | {q_hung['mean_similarity']:.4f} | {q_hung['n_pairs']} | {q_hung['collision_rate']:.3f} |

Hungarian gain over greedy baseline: **{gain:+.4f}** mean similarity.
Collisions (multiple predictions claiming one segment) drop from
{q_greedy['n_collisions']} to {q_hung['n_collisions']}.
"""
    (OUT / "results.md").write_text(md)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(["greedy argmax\n(Hungarian OFF)", "Hungarian assignment\n(ON)"],
           [q_greedy["mean_similarity"], q_hung["mean_similarity"]])
    ax.set_ylabel("mean matched-pair cosine similarity")
    ax.set_title("Content-aware region matching: ablation of Hungarian assignment")
    ax.set_ylim(0, 1)
    for i, v in enumerate([q_greedy["mean_similarity"], q_hung["mean_similarity"]]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(OUT / "charts.png", dpi=120)
    plt.close(fig)
    print(f"wrote {OUT / 'results.md'} and {OUT / 'charts.png'}")
    return {"greedy": q_greedy, "hungarian": q_hung, "gain": gain}


if __name__ == "__main__":
    main()
