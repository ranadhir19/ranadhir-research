"""Quickstart: type-aware correlate() facade (runs in seconds).

One planted association per dispatch branch on a small synthetic table,
plus a quick null-pair check.
"""

import numpy as np

from ghosh_research.bukpi import correlate, pearson_r

rng = np.random.default_rng(2)
n = 300

num_a = rng.standard_normal(n)
num_b = 0.7 * num_a + np.sqrt(1 - 0.49) * rng.standard_normal(n)

groups = rng.choice(["a", "b"], size=n)
num_c = np.where(groups == "a", -1.0, 1.0) + rng.standard_normal(n)

cat_e = rng.choice(["X", "Y"], size=n)
cat_f = np.where(rng.random(n) < 0.8, cat_e, rng.choice(["X", "Y"], size=n))

texts = ["product review: excellent, superb, fantastic." if s == 0
         else "product review: terrible, awful, dreadful." for s in rng.choice([0, 1], size=n)]
labels = np.array([0 if "excellent" in t else 1 for t in texts])

for name, x, y in [("numeric-numeric", num_a, num_b),
                   ("numeric-categorical", num_c, groups),
                   ("categorical-categorical", cat_e, cat_f)]:
    r = correlate(x, y)
    print(f"{name:24s} -> {r['measure']}: {r['value']:.3f}")
r = correlate(texts, labels, rng)
print(f"{'text vs label':24s} -> {r['measure']}: {r['value']:.3f}")

null = [abs(pearson_r(rng.standard_normal(n), rng.standard_normal(n))) for _ in range(20)]
print(f"null pairs (20): mean|r| = {np.mean(null):.3f}, max|r| = {np.max(null):.3f}")
