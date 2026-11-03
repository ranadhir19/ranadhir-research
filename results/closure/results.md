# Closure experiment results

Config: 2000 steps x 30 seeds, p_novel=0.3, alpha=0.05,
novelty test |z| > 1.9600 under the N(0,1) prior.

| agent | measured novel rate | 95% CI | theoretical |
|---|---|---|---|
| pure AI (regime 0 only) | 0.0483 | +- 0.0019 | 0.0500 |
| hybrid (30% novel-regime obs.) | 0.3260 | +- 0.0034 | 0.3288 |

The pure agent's novel rate sits at the significance level alpha: confined to
its prior, it cannot transcend it. The hybrid agent's rate matches the
mixture prediction p * power + (1-p) * alpha with detection power
0.9793 on the N(4,1) novel regime.
