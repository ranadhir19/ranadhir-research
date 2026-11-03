"""siamese — contrastive verification for bank-cheque CAR/LAR amounts.

Original implementation of the core idea of "Siamese Networks for
Bank-Cheque CAR/LAR Verification" (under journal revision, 2026):
synthetic CAR/LAR pair generation, bigram-hash character features, and a
linear projection trained with contrastive loss, compared against a
raw-feature-distance baseline.
"""

from .siamese import (
    num_words,
    make_car,
    make_lar,
    bigram_features,
    pair_vector,
    generate_pairs,
    train_projection,
    embed_distances,
    accuracy_at_threshold,
    best_threshold,
    train_linear_classifier,
    linear_predict,
)

__all__ = [
    "num_words",
    "make_car",
    "make_lar",
    "bigram_features",
    "pair_vector",
    "generate_pairs",
    "train_projection",
    "embed_distances",
    "accuracy_at_threshold",
    "best_threshold",
    "train_linear_classifier",
    "linear_predict",
]
