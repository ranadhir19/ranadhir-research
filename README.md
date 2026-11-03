# ghosh-research

Working applications implementing the core ideas of eleven research papers by Ranadhir Ghosh.
Each paper becomes a runnable, self-contained system — CPU only (numpy / scipy / matplotlib) —
with a small experiment producing measured results and a chart.

## Watch it built

The papers behind these applications are taken apart on camera at **[The Architect's AI Workshop](https://www.youtube.com/@ArchitectsAIWorkshop)** — *first principles on the whiteboard, built live in the editor, measured honestly.* Browse the episodes: [youtube.com/@ArchitectsAIWorkshop/videos](https://www.youtube.com/@ArchitectsAIWorkshop/videos)

## The eleven applications

| # | Package | Paper | What it does | Headline result |
|---|---------|-------|--------------|-----------------|
| 01 | `ghosh_research.cst` | Coarse Set Theory for Ethical AI | Greedy maximal ethical subset construction | Demographic parity 0.1327 -> 0.0148 at 1.8pp accuracy cost |
| 02 | `ghosh_research.cais` | Semantic Chain-of-Thought CAIS | Hungarian assignment for document regions | 0.9277 vs 0.8605 greedy matching similarity |
| 03 | `ghosh_research.closure` | The Agent Closure Theorem | Hybrid human-AI discovery simulation | Hybrid novel rate 0.326 vs pure-AI 0.048 |
| 04 | `ghosh_research.pinn` | PINNs for Protein LLPS Prediction | Hard-negative stress test + physics-informed loss | AUC 1.000 -> 0.953 -> 0.906 across P1/P2/P3 strata |
| 05 | `ghosh_research.dsai` | Theoretical Augmentation for Option Pricing | Black-Scholes + neural residual, tested in a crisis regime | Crisis RMSE 3.652 (hybrid) vs 4.730 (pure ML) |
| 06 | `ghosh_research.dynamo` | DYNAMO: Dynamic Adaptive Neural Architecture | Contractive world-model operator verification | Empirical contraction below predicted rho; tight at L=0.5 (0.824 vs 0.850) |
| 07 | `ghosh_research.kpi` | Bayesian Validation of Concept-Map Edges | MH sampler + Savage-Dickey Bayes factors | 9/12 edges within 9%; null edge BF01 = 53.5 |
| 08 | `ghosh_research.bukpi` | Type-Aware Bottom-Up KPI Discovery | Type-dispatching correlate() facade | All 4 branches recovered; null mean|r| = 0.034 |
| 09 | `ghosh_research.memento` | Multi-Agent Memento | Retention/plasticity trade-off under shift | Memory 0.715 overall vs refit 0.567; refit wins late segment 0.927 |
| 10 | `ghosh_research.siamese` | Siamese Networks for Cheque CAR/LAR Verification | Contrastive amount-vs-words matching | Embedding 0.980 vs linear 0.508 (chance) |
| 11 | `ghosh_research.cardio` | Preliminary investigation: forecasting cardiovascular risk from the resting ECG | Null-calibrated horizon durability of ECG risk (synthetic) | IPA rise = artefact (in null range); concordance fall = real (outside null range) |

All numbers are measured from actual runs (seeds recorded in `results/<slug>/results.md`).
Each project's README cites its paper and documents deviations where a clean synthetic
benchmark could not reproduce a paper figure — the honest numbers are reported, not fabricated ones.

## Install

Requires Python 3.10+.

```bash
git clone https://github.com/ranadhir19/research.git
cd research
pip install numpy scipy matplotlib
```

(`pip install -e .` works on systems without PEP 668 enforcement; on PEP 668 systems,
use `PYTHONPATH=src` for the examples, e.g. `PYTHONPATH=src python examples/06_dynamo_quickstart.py`.)

## Quickstart

```bash
# run a one-minute demo of any application
PYTHONPATH=src python examples/01_cst_quickstart.py

# or use the library directly
PYTHONPATH=src python - <<'EOF'
from ghosh_research.dynamo import contractive_operator, iterate, spectral_clip
import numpy as np
rng = np.random.default_rng(0)
T, rho = contractive_operator(64, 0.5, alpha=0.3, rng=rng)  # rho = 0.85 guaranteed
x = rng.standard_normal(64)
for k in range(50):
    x = T(x)
print("predicted rho:", rho)
EOF
```

## Layout

```
src/ghosh_research/<slug>/   one documented package per application
experiments/<slug>_experiment.py   full experiment scripts (the source of every reported number)
results/<slug>/{results.md,charts.png}   measured tables and charts
<slug>/README.md              per-project README citing its paper
examples/NN_<slug>_quickstart.py   runnable one-minute demos
docs/                         static landing site (GitHub Pages) + user-guide.md
```

## Documentation

- [User guide](docs/user-guide.md) — how to run, extend, and reproduce each application
- [Landing site](https://ranadhir19.github.io/research/) — one page per application:
  Problem / The paper / The implementation / Architecture / Results / Insights / Code
- [PyPI release checklist](PYPI_RELEASE.md) — distribution name `ghosh-research` (verified free);
  the package is not published.

## Notes

- Implementations are original code expressing each paper's ideas; no paper text is copied
  beyond fair-use titles and short descriptions. Each paper is under journal revision.
- Author: Ranadhir Ghosh. MIT License.
