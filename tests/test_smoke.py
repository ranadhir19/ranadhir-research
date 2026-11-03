"""Smoke tests for ghosh-research (seeded, deterministic, offline).

This suite was written for the validate.py harness. It verifies, for each
of the eleven paper applications:

* the subpackage imports and exposes its documented public API, and
* the app's quickstart example (examples/NN_<slug>_quickstart.py) runs
  end-to-end in its seeded mode and its printed output satisfies the
  qualitative claim documented in that quickstart's own docstring
  (e.g. Hungarian assignment beats greedy; the hybrid pricer beats pure
  ML in the crisis regime; empirical contraction stays below the
  predicted rho).

It does NOT re-run the full experiments in experiments/ and does NOT
verify the headline numbers in README.md / results/<slug>/results.md
(those come from the full experiments, with different sample sizes).
See VALIDATION.md for the honest coverage statement.

Apps with a runnable entry point: all eleven (via examples/).
IMPORT_ONLY_APPS lists any app checked by import only; it is currently
empty and validate.py prints it on every run.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

SLUGS = [
    "cst", "cais", "closure", "pinn", "dsai", "dynamo",
    "kpi", "bukpi", "memento", "siamese", "cardio",
]

# Apps without a runnable entry point (import-only checks). None today:
# every app ships examples/NN_<slug>_quickstart.py.
IMPORT_ONLY_APPS = []

EXAMPLES = {
    "cst": "01_cst_quickstart.py",
    "cais": "02_cais_quickstart.py",
    "closure": "03_closure_quickstart.py",
    "pinn": "04_pinn_quickstart.py",
    "dsai": "05_dsai_quickstart.py",
    "dynamo": "06_dynamo_quickstart.py",
    "kpi": "07_kpi_quickstart.py",
    "bukpi": "08_bukpi_quickstart.py",
    "memento": "09_memento_quickstart.py",
    "siamese": "10_siamese_quickstart.py",
    "cardio": "11_cardio_quickstart.py",
}


def _floats(pattern, text):
    m = re.search(pattern, text)
    assert m, f"pattern {pattern!r} not found in output:\n{text}"
    return [float(g) for g in m.groups()]


def _check_cst(out):
    base, = _floats(r"baseline: DP gap = ([\d.]+)", out)
    adm, = _floats(r"decisions: DP gap = ([\d.]+)", out)
    assert adm < base, "ethical subset did not reduce the DP gap"
    assert adm <= 0.015 + 1e-9, "admitted DP gap above the tau_dp=0.015 target"


def _check_cais(out):
    greedy, = _floats(r"greedy argmax:\s+mean similarity = ([\d.]+)", out)
    hung, coll = _floats(r"Hungarian:\s+mean similarity = ([\d.]+), collisions = (\d+)", out)
    assert hung > greedy, "Hungarian assignment did not beat greedy"
    assert coll == 0, "Hungarian assignment produced collisions"


def _check_closure(out):
    pure, theory_pure = _floats(r"pure AI agent: novel rate = ([\d.]+) \(theory: ([\d.]+)\)", out)
    hybrid, _ = _floats(r"hybrid agent:\s+novel rate = ([\d.]+) \(theory: ([\d.]+)\)", out)
    assert pure <= theory_pure + 0.01, "pure agent exceeded the closure bound"
    assert hybrid > 4 * pure, "hybrid agent did not clearly break the bound"


def _check_pinn(out):
    p1, = _floats(r"P1: pos vs easy: AUC = ([\d.]+)", out)
    p3, = _floats(r"P3: pos vs hard: AUC = ([\d.]+)", out)
    assert p1 >= 0.99, "easy-negative protocol AUC unexpectedly low"
    assert p3 < p1, "hard negatives did not degrade AUC (stress test inert)"


def _check_dsai(out):
    pure, = _floats(r"pure ML\s+: crisis RMSE = ([\d.]+)", out)
    hybrid, = _floats(r"hybrid \(BS \+ residual\): crisis RMSE = ([\d.]+)", out)
    assert hybrid < pure, "hybrid did not beat pure ML in the crisis regime"


def _check_dynamo(out):
    assert "Theorem holds" in out
    rho, emp_max = _floats(r"predicted rho = ([\d.]+); empirical mean = [\d.]+, max = ([\d.]+)", out)
    assert emp_max <= rho + 1e-9


def _check_kpi(out):
    beta, = _floats(r"true edge:\s+beta_hat = ([\d.]+)", out)
    assert abs(beta - 1.5) < 0.15, "true-edge estimate far from the planted 1.5"
    assert "Null edge correctly rejected" in out


def _check_bukpi(out):
    r, = _floats(r"numeric-numeric\s+-> pearson_r: ([\d.]+)", out)
    assert r > 0.5, "planted numeric association not recovered"
    v, = _floats(r"text vs label\s+-> cramers_v\(text_clusters, label\): ([\d.]+)", out)
    assert v > 0.9, "text branch did not recover the planted label association"
    mean_null, = _floats(r"null pairs \(20\): mean\|r\| = ([\d.]+)", out)
    assert mean_null < 0.15, "null pairs show spurious correlation"


def _check_memento(out):
    refit, = _floats(r"refit \(plastic\) : overall=([\d.]+)", out)
    memory, = _floats(r"memory \(retains\): overall=([\d.]+)", out)
    assert memory > refit, "memory bank did not beat refit overall"


def _check_siamese(out):
    lin, = _floats(r"linear classifier:\s+([\d.]+)", out)
    sia, = _floats(r"siamese embedding:\s+([\d.]+)", out)
    assert sia > 0.9, "siamese embedding accuracy unexpectedly low"
    assert sia > lin + 0.3, "siamese embedding did not clearly beat linear"


def _check_cardio(out):
    assert "all p > 0.05: True" in out
    n, = _floats(r"knowledge-graph tensions detected: (\d+)", out)
    assert n == 1, "expected exactly one planted claim-graph tension"


CHECKS = {
    "cst": _check_cst, "cais": _check_cais, "closure": _check_closure,
    "pinn": _check_pinn, "dsai": _check_dsai, "dynamo": _check_dynamo,
    "kpi": _check_kpi, "bukpi": _check_bukpi, "memento": _check_memento,
    "siamese": _check_siamese, "cardio": _check_cardio,
}


def test_package_and_subpackage_imports():
    import ghosh_research
    assert sorted(ghosh_research.__all__) == sorted(SLUGS)
    for slug in SLUGS:
        mod = __import__(f"ghosh_research.{slug}", fromlist=["__all__"])
        assert mod.__all__, f"{slug} exposes an empty __all__"
        for name in mod.__all__:
            assert hasattr(mod, name), f"ghosh_research.{slug}.{name} missing"


def _run_quickstart(slug):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "examples" / EXAMPLES[slug])],
        capture_output=True, text=True, timeout=300, env=env, cwd=str(ROOT))
    assert proc.returncode == 0, (
        f"{EXAMPLES[slug]} exited {proc.returncode}\n{proc.stderr[-2000:]}")
    CHECKS[slug](proc.stdout)


def test_quickstart_cst():
    _run_quickstart("cst")


def test_quickstart_cais():
    _run_quickstart("cais")


def test_quickstart_closure():
    _run_quickstart("closure")


def test_quickstart_pinn():
    _run_quickstart("pinn")


def test_quickstart_dsai():
    _run_quickstart("dsai")


def test_quickstart_dynamo():
    _run_quickstart("dynamo")


def test_quickstart_kpi():
    _run_quickstart("kpi")


def test_quickstart_bukpi():
    _run_quickstart("bukpi")


def test_quickstart_memento():
    _run_quickstart("memento")


def test_quickstart_siamese():
    _run_quickstart("siamese")


def test_quickstart_cardio():
    _run_quickstart("cardio")
