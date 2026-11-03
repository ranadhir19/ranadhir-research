"""ghosh_research — runnable implementations of Ranadhir Ghosh's research papers.

Each subpackage implements the core technical contribution of one paper
as a self-contained, reproducible application:

- cst:       Coarse Set Theory — maximal ethical subset construction
- cais:      Semantic CoT-CAIS — content-aware document region matching
- closure:   Agent Closure Theorem — hybrid discovery simulation
- pinn:      Physics-informed LLPS protein classification + hard-negative stress test
- dsai:      Theory-augmented (Black-Scholes + ML residual) option pricing
- dynamo:    Contractive world-model operator verification
- kpi:       Bayesian KPI governance — MH sampling + Savage-Dickey Bayes factors
- bukpi:     Bottom-up type-aware KPI discovery facade
- memento:   Continual learning — retention vs plasticity trade-off
- siamese:   Siamese contrastive verification for cheque CAR/LAR amounts
- cardio:    Horizon durability of ECG-derived cardiovascular risk
"""

__version__ = "0.1.0"

__all__ = [
    "cst", "cais", "closure", "pinn", "dsai",
    "dynamo", "kpi", "bukpi", "memento", "siamese",
    "cardio",
]
