# BUKPI — type-aware correlate() facade

Seed: 20261019; n = 500 rows; text branch uses 64-d hash sketches + Lloyd k-means (k=3).

| branch | planted strength | measure | recovered | status |
|--------|------------------|---------|-----------|--------|
| numeric-numeric (Pearson r) | ~0.70 | pearson_r | 0.728 | recovered |
| numeric-categorical (ANOVA eta^2) | ~0.50 | anova_eta2 | 0.561 | recovered |
| categorical-categorical (Cramer's V) | ~0.60 | cramers_v | 0.713 | recovered |
| text vs sentiment (cluster Cramer's V) | ~0.95 | cramers_v(text_clusters, label) | 1.000 | recovered |

## Null-pair robustness (100 independent numeric pairs)

- mean |r| = **0.0344** (expect ~= 0.03)
- max |r| = **0.1006** (expect < 0.2): **PASS**

All planted associations recovered: **True**.
