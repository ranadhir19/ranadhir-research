# Validation — ghosh-research

Harness: `python3 validate.py` (repo root). Deterministic, offline.
Exit 0 and a final line of `HARNESS GREEN` mean every stage passed.

## Stages

1. **Import/compile** — `compileall` over `src/`, `tests/`, `examples/`
   and `experiments/`, then a clean subprocess imports the package named
   in `pyproject.toml` (`ghosh-research` → `ghosh_research`) plus all 11
   subpackages.
2. **Test suite** — `pytest tests/` in a subprocess (see below).
3. **Employer-reference scan** — the standing gate scanner
   (the workspace reference scanner) over the whole tree; every candidate is judged in
   context by `validate.py`. A real reference fails the stage; an
   affiliation mention in research content would be labelled
   USER-DECISION and would also fail — it is the owner's call, never
   silently removed by tooling.
4. **Results/docs consistency** — every `results/…`, `examples/…`,
   `experiments/…`, `docs/…` path referenced by the root and per-app
   READMEs exists; placeholder paths (`results/<slug>/…`,
   `examples/NN_<slug>_quickstart.py`) are expanded over the 11 app
   slugs. Existence only — file contents are not compared.
   Stage 4 also includes **deployment-config validation**: the tree is
   scanned for `render.yaml`/`render.yml`, `vercel.json`, `Procfile`,
   `Dockerfile*`, `docker-compose*.yml`/`yaml`, `netlify.toml`,
   `fly.toml`, and `.github/workflows/*.yml`/`yaml`. Any file found is
   parsed and schema-checked (e.g. a Render service with `type: static`
   fails — a static site is `type: web` with `env: static`); a broken
   deploy config fails the stage. Current result: **no deployment
   configs present as of this run** — the step passes with nothing to
   validate.

## What the test suite verifies

`tests/test_smoke.py` (12 tests, added with this harness — the repo
previously had no tests at all):

- The top-level package's `__all__` names exactly the 11 apps, every
  subpackage imports, and every name in each subpackage's `__all__`
  resolves.
- Each app's quickstart (`examples/NN_<slug>_quickstart.py`) runs
  end-to-end with its fixed seed, exits 0, and its printed output
  satisfies the qualitative claim in that quickstart's own docstring,
  checked as inequalities on the parsed numbers (Hungarian > greedy for
  cais; admitted DP gap below the 0.015 target and below baseline for
  cst; hybrid crisis RMSE < pure-ML for dsai; empirical contraction ≤
  predicted rho for dynamo; null edge rejected for kpi; siamese
  embedding > 0.9 and > linear + 0.3; exactly one planted claim-graph
  tension for cardio; etc.).

**Import-only apps: none.** All 11 apps have a runnable quickstart and
are exercised through it. `validate.py` prints this list on every run.

## What is NOT verified (honest non-coverage)

- **The headline numbers are not re-derived.** The figures in
  `README.md` and `results/<slug>/results.md` come from the full
  experiments in `experiments/`; the harness only compile-checks those
  scripts, it does not run them. Quickstart runs use smaller samples
  and different seeds, so quickstart numbers legitimately differ from
  the published experiment numbers — the smoke suite asserts the
  documented *behaviour*, not the published *values*.
- **Scientific correctness** of the implementations relative to the
  papers is not verified; neither is the fidelity of each app to its
  paper beyond the quickstart-level claims above.
- **Charts** (`results/<slug>/charts.png`, docs assets) are checked for
  existence only; image contents are never inspected.
- **The docs site** (`docs/**/*.html`) is not rendered or link-checked;
  only README-referenced paths are existence-checked.
- **Packaging** (`pip install`, PyPI metadata beyond the project name)
  is not exercised.
- **The scan** covers text files only (the scanner skips binaries) and
  excludes `validate.py` and this file, which must name the gate's
  patterns to enforce them; those self-references (pattern names, the
  scanner's own path and its override variable) were judged benign in
  context — they describe the gate, they are not content references.
  The harness prints how many such excluded candidates it saw.

## Provenance

Tests authored 2026-10-20 alongside the harness, from the quickstart
docstrings and the package's documented public API — not from the
experiment scripts' internals.
