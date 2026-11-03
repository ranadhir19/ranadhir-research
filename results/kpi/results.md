# KPI — Bayesian concept-map edge validation

Seed: 20261019; n = 300 per edge; MH 25000 iterations (8000 burn-in); prior beta ~ N(0, 10^2).

| edge | beta* | beta_hat | rel. err | 95% HDI | covers beta* | BF10 | BF01 |
|------|-------|----------|----------|---------|----------------|------|------|
| revenue ~ ad_spend | 0.800 | 0.8393 | 4.91% | [0.8116, 0.8683] | False | >1e300 | 0 |
| churn ~ latency | -1.500 | -1.5667 | 4.45% | [-1.6484, -1.4866] | True | 7.41e+300 | 1.35e-301 |
| margin ~ discount | -22.000 | -22.7821 | 3.56% | [-23.8981, -21.6527] | True | >1e300 | 0 |
| nps ~ response_hrs | -0.350 | -0.3688 | 5.38% | [-0.4137, -0.3246] | True | 9.16e+57 | 1.09e-58 |
| uptime ~ load | 0.060 | 0.0520 | 13.26% | [0.0422, 0.0614] | True | 2.54e+21 | 3.94e-22 |
| conversion ~ adstock | 2.400 | 2.4433 | 1.80% | [2.2369, 2.6436] | True | 4.98e+118 | 2.01e-119 |
| retention ~ tickets | -0.900 | -0.8856 | 1.60% | [-0.9387, -0.8311] | True | 7.1e+223 | 1.41e-224 |
| cost ~ volume | 1.100 | 1.0135 | 7.86% | [0.9249, 1.1058] | True | 1.99e+104 | 5.02e-105 |
| csat ~ wait_min | -0.120 | -0.1058 | 11.82% | [-0.1226, -0.0888] | True | 4.85e+29 | 2.06e-30 |
| growth ~ hiring | 0.450 | 0.4307 | 4.29% | [0.3668, 0.4924] | True | 3.7e+36 | 2.7e-37 |
| risk ~ leverage | 3.300 | 3.2043 | 2.90% | [2.9798, 3.4326] | True | 3.12e+167 | 3.21e-168 |
| output ~ overtime | 0.020 | 0.0134 | 33.02% | [0.0066, 0.0202] | True | 0.936 | 1.07 |
| null: noise ~ noise | 0.000 | 0.0914 | n/a | [-0.0242, 0.2119] | True | 0.0187 | 53.5 |

- Edges recovered within 9% relative error: **9/12**.
- Central-95% HDI coverage of beta*: **92%** (12/13).
- Null edge BF01 = **53.5** (strong evidence for the null: the absent edge is correctly rejected).
