#!/usr/bin/env python3
"""Probe 6 -- Cycle 2 instantiation: certification-budget audit of a real
pipeline artifact (agentic-systems-lab system 16, enriched-observability).

Pre-registered in Paper_02_build_note.md section 9.3 BEFORE this run.
Deterministic: only the pipeline's own seeded generators (literal seeds),
literal threshold 3.0 (the pipeline's shipped default), no hash(), no
wall-clock, no randomness of this probe's own.

Artifact: ~/workspace/agentic-systems-lab
  src/agentic_systems/enriched_observability/{telemetry,enrich,__init__}.py
  telemetry.simulate(seed); enrich.robust_z(series, window=60);
  per-minute flag |z| >= 3.0 with warmup t >= 30 (the per-minute flag
  underlying enrich.detect_intervals; its min_len=5 interval grouping
  is NOT applied -- certification here is per candidate minute).
Seeds (the pipeline harness's own SEEDS list, split in list order):
  calibration [7, 11]; audit [42, 0, 99].
Outputs: probe6_output.txt (via stdout capture by the runner) and
  probe6_records.csv (candidate-level records).
"""
from __future__ import annotations

import csv
import math
import os
import sys

sys.path.insert(0, os.path.expanduser("~/workspace/agentic-systems-lab/src"))

import numpy as np

from agentic_systems.enriched_observability import LAYER_METRICS, telemetry
from agentic_systems.enriched_observability import enrich
from agentic_systems.enriched_observability import _get_series

HERE = os.path.dirname(os.path.abspath(__file__))
CAL_SEEDS = [7, 11]
AUDIT_SEEDS = [42, 0, 99]
Z_THRESH = 3.0
WARMUP = 30
T_MIN = telemetry.T_MIN


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def gt_mask(anomalies, metric_name: str) -> np.ndarray:
    m = np.zeros(T_MIN, dtype=bool)
    for s, e, metric, _label in anomalies:
        if metric == metric_name:
            m[s:min(e, T_MIN)] = True
    return m


def main() -> int:
    # ---- load every (seed, layer): z-series, flags, gt masks -------------
    # store[seed][layer] = dict(metric_names=[...], flags={name: bool[T]},
    #                           clean={name: bool[T]}, absz={name: float[T]})
    store: dict = {}
    for seed in CAL_SEEDS + AUDIT_SEEDS:
        data = telemetry.simulate(seed)
        store[seed] = {}
        for layer, metrics in LAYER_METRICS.items():
            names, flags, clean, absz = [], {}, {}, {}
            anomalies = data[layer]["anomalies"]
            for key, sub in metrics:
                name = key if sub is None else f"{key}[{sub}]"
                s = _get_series(data, layer, key, sub)
                z = enrich.robust_z(s)
                az = np.abs(z)
                f = az >= Z_THRESH
                f[:WARMUP] = False
                names.append(name)
                flags[name] = f
                absz[name] = az
                clean[name] = ~gt_mask(anomalies, name)
            store[seed][layer] = dict(names=names, flags=flags, clean=clean, absz=absz)
        print(f"loaded seed {seed}")

    checks = [(layer, name) for layer, ms in LAYER_METRICS.items()
              for name in store[CAL_SEEDS[0]][layer]["names"]]

    # ---- (a) per-check alpha-hat on calibration clean minutes ------------
    print("\n=== (a) per-check empirical false-flag rates, calibration seeds "
          f"{CAL_SEEDS} (threshold {Z_THRESH}, shipped default) ===")
    alpha: dict = {}
    b_layer: dict = {}
    b_lo: dict = {}
    b_hi: dict = {}
    for layer, name in checks:
        k = n = 0
        for seed in CAL_SEEDS:
            st = store[seed][layer]
            cm = st["clean"][name].copy()
            cm[:WARMUP] = False
            n += int(cm.sum())
            k += int((st["flags"][name] & cm).sum())
        p, lo, hi = wilson(k, n)
        alpha[(layer, name)] = p
        b_layer[layer] = b_layer.get(layer, 0.0) + p
        b_lo[layer] = b_lo.get(layer, 0.0) + lo
        b_hi[layer] = b_hi.get(layer, 0.0) + hi
        print(f"alpha_hat[{layer}/{name}] = {p:.4f}  Wilson95=[{lo:.4f},{hi:.4f}]"
              f"  flags={k} clean_n={n}")
    print("\n=== (b) per-layer total budget B_hat = sum_j alpha_hat_j ===")
    for layer in LAYER_METRICS:
        print(f"B_hat[{layer}] = {b_layer[layer]:.4f}  "
              f"sum-of-Wilson95=[{b_lo[layer]:.4f},{b_hi[layer]:.4f}]  "
              f"m={len(LAYER_METRICS[layer])}")
    mean_b = float(np.mean([b_layer[l] for l in LAYER_METRICS]))
    print(f"mean per-layer B_hat = {mean_b:.4f}")

    # ---- candidate-level records + audit streams --------------------------
    rows = []
    # audit per layer: list of (union_flag, first_flag, sel_name, sel_absz)
    audit: dict = {layer: [] for layer in LAYER_METRICS}
    for seed in CAL_SEEDS + AUDIT_SEEDS:
        split = "calibration" if seed in CAL_SEEDS else "audit"
        for layer, metrics in LAYER_METRICS.items():
            st = store[seed][layer]
            names = st["names"]
            for t in range(WARMUP, T_MIN):
                union_clean = all(bool(st["clean"][nm][t]) for nm in names)
                fl = [1 if st["flags"][nm][t] else 0 for nm in names]
                azs = [float(st["absz"][nm][t]) for nm in names]
                jstar = int(np.argmax(azs))
                rows.append([seed, split, layer, t, 1 if union_clean else 0,
                             ";".join(str(x) for x in fl), 1 if any(fl) else 0,
                             fl[0], names[jstar], f"{azs[jstar]:.6f}"])
                if split == "audit" and union_clean:
                    audit[layer].append((1 if any(fl) else 0, fl[0],
                                         names[jstar], azs[jstar]))
    with open(os.path.join(HERE, "probe6_records.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "split", "layer", "minute", "union_clean",
                    "flags_per_check", "union_flag", "first_check_flag",
                    "selected_check_max_absz", "selected_absz"])
        w.writerows(rows)
    print(f"\ncandidate-level records written: {len(rows)} rows -> probe6_records.csv")

    # ---- (c) sequential audit per layer -----------------------------------
    print(f"\n=== (c) sequential audit, audit seeds {AUDIT_SEEDS} in order, "
          "union-clean candidates in chronological order ===")
    tot_v = tot_n = 0
    tot_b = 0.0
    for layer in LAYER_METRICS:
        seq = audit[layer]
        n = len(seq)
        b = b_layer[layer]
        v = 0
        max_excess = -1e18
        for i, (uf, *_rest) in enumerate(seq, start=1):
            v += uf
            max_excess = max(max_excess, v - i * b)
        slack = math.sqrt(n * math.log(1 / 0.05) / 2) if n else 0.0
        bn = n * b
        ok = v <= bn + slack
        tot_v += v; tot_n += n; tot_b += bn
        print(f"audit[{layer}]: N={n} V_N={v} B_N={bn:.4f} V_N-B_N={v-bn:+.4f} "
              f"max_t(V_t-B_t)={max_excess:+.4f} Azuma_slack_delta0.05={slack:.4f} "
              f"V_N<=B_N+slack: {ok} empirical_union_rate={v/n:.4f}")
    slack_tot = math.sqrt(tot_n * math.log(1 / 0.05) / 2)
    print(f"audit[POOLED layers as separate streams, totals]: N={tot_n} V_N={tot_v} "
          f"B_N={tot_b:.4f} V_N-B_N={tot_v-tot_b:+.4f}")

    # ---- (d) Proposition 4.7 regimes on audit data -------------------------
    print("\n=== (d) candidate-dependent regimes on audit data (union-clean) ===")
    for scope, layers in [("POOLED", list(LAYER_METRICS))] + \
                         [(l, [l]) for l in LAYER_METRICS]:
        cand = [c for l in layers for c in audit[l]]
        n = len(cand)
        if n == 0:
            continue
        f_rate = float(np.mean([c[1] for c in cand]))
        f_nom = float(np.mean([alpha[(l, audit_first(l))] for l in layers])) \
            if scope != "POOLED" else float(np.mean(
                [alpha[(l, store[AUDIT_SEEDS[0]][l]["names"][0])]
                 for l in layers for _ in audit[l]]))
        s_rate = float(np.mean([c[0] for c in cand]))
        s_nom = float(np.mean([b_layer[l] for l in layers])) if scope != "POOLED" \
            else float(np.mean([b_layer[l] for l in layers for _ in audit[l]]))
        c_nom = float(np.mean([alpha[(l, c[2])] for l in layers for c in audit[l]])) \
            if scope != "POOLED" else float(np.mean(
                [alpha[(l, c[2])] for l in layers for c in audit[l]]))
        # regime C flags by construction: |z_j*| >= |z_j*|
        c_rate = 1.0
        print(f"regimes[{scope}]: n={n} "
              f"F_fixed_first_check nominal={f_nom:.4f} actual={f_rate:.4f} | "
              f"S_selected_fixed_threshold nominal_B={s_nom:.4f} actual={s_rate:.4f} | "
              f"C_constructed_threshold nominal={c_nom:.4f} actual={c_rate:.4f}")
    print("\nDONE probe6")
    return 0


def audit_first(layer: str) -> str:
    return LAYER_METRICS[layer][0][0] if LAYER_METRICS[layer][0][1] is None else \
        f"{LAYER_METRICS[layer][0][0]}[{LAYER_METRICS[layer][0][1]}]"


if __name__ == "__main__":
    raise SystemExit(main())
