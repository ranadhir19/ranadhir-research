"""Quickstart: siamese CAR/LAR verification (runs in seconds).

Trains the 16-d contrastive projection on 800 synthetic cheque pairs and
compares the siamese embedding against the raw-distance baseline and a
purely linear classifier on 300 test pairs.
"""

import numpy as np

from ghosh_research.siamese import (
    generate_pairs,
    train_projection,
    embed_distances,
    accuracy_at_threshold,
    best_threshold,
    train_linear_classifier,
    linear_predict,
)

rng = np.random.default_rng(4)
Xtr, ytr = generate_pairs(rng, 800)
Xte, yte = generate_pairs(rng, 300)

thr = best_threshold(np.linalg.norm(Xtr, axis=1), ytr)
acc_raw = accuracy_at_threshold(np.linalg.norm(Xte, axis=1), yte, thr)

w, b = train_linear_classifier(Xtr, ytr, rng, epochs=40)
acc_lin = float((linear_predict(Xte, w, b) == yte).mean())

W = train_projection(Xtr, ytr, rng, epochs=15)
thr_e = best_threshold(embed_distances(Xtr, W), ytr)
acc_sia = accuracy_at_threshold(embed_distances(Xte, W), yte, thr_e)

print(f"raw-distance baseline: {acc_raw:.3f}")
print(f"linear classifier:     {acc_lin:.3f}")
print(f"siamese embedding:     {acc_sia:.3f}")
print("Linear methods sit near chance; the contrastive embedding cracks clean synthetic pairs.")
