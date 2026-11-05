#!/usr/bin/env python3>
"""Component C algebra verification, per upgrade2_preregistration.md
Section 2. Population part: exact analytic covariances. Empirical part:
brute-force sample covariances at n = 1,000,000. Output: algebra_results.json
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def adj_slope_from_cov(S):
    """Coefficient of x (index 0) in the population linear projection of
    y (index 2) on (x, z-index-1), from a 3x3 covariance matrix in the
    order (x, z, y) WITHOUT intercept (means are zero)."""
    Sxx = S[:2, :2]
    Sxy = S[:2, 2]
    coef = np.linalg.solve(Sxx, Sxy)
    return float(coef[0])


def population_cov(a, beta, gamma, sd_e, lam):
    """DGP: z ~ N(0,1); x = a z + sqrt(1-a^2) ex; y = beta x + gamma z
    + sd_e ey; u var = (1-lam)/lam; z* = z + u. Returns (true cov of
    (x,z,y), standardised observed cov of (x,z*,y))."""
    var_z = 1.0
    var_x = a * a * var_z + (1 - a * a)
    cov_xz = a * var_z
    # y = beta x + gamma z + e  -> express in (z, ex, ey)
    # y = (beta a + gamma) z + beta sqrt(1-a^2) ex + sd_e ey
    cy_z = beta * a + gamma
    var_y = cy_z ** 2 + beta ** 2 * (1 - a * a) + sd_e ** 2
    cov_xy = beta * var_x + gamma * cov_xz
    cov_zy = gamma * var_z + beta * cov_xz
    S_true = np.array([[var_x, cov_xz, cov_xy],
                       [cov_xz, var_z, cov_zy],
                       [cov_xy, cov_zy, var_y]])
    var_u = (1 - lam) / lam * var_z
    var_zs = var_z + var_u
    S_obs_raw = np.array([[var_x, cov_xz, cov_xy],
                          [cov_xz, var_zs, cov_zy],
                          [cov_xy, cov_zy, var_y]])
    d = np.sqrt(np.diag(S_obs_raw))
    S_obs = S_obs_raw / np.outer(d, d)
    d_t = np.sqrt(np.diag(S_true))
    S_true_std = S_true / np.outer(d_t, d_t)
    return S_true_std, S_obs


def corrected_slope(S_obs_std, lam):
    S = S_obs_std.copy()
    S[1, 1] = lam * S_obs_std[1, 1]   # Var(z*) replaced by lam * Var(z*)
    return adj_slope_from_cov(S)


def main():
    rng = np.random.default_rng(2026102702)
    lams = [0.5, 0.64, 0.8, 0.9, 1.0]
    rows, max_disc = [], 0.0
    for i in range(500):
        a = rng.uniform(-0.85, 0.85)
        beta = rng.uniform(-0.8, 0.8)
        gamma = rng.uniform(-1.0, 1.0)
        sd_e = rng.uniform(0.5, 1.5)
        lam = lams[i % len(lams)]
        S_true, S_obs = population_cov(a, beta, gamma, sd_e, lam)
        true_slope = adj_slope_from_cov(S_true)
        corr_slope = corrected_slope(S_obs, lam)
        disc = abs(true_slope - corr_slope)
        max_disc = max(max_disc, disc)
        rows.append({"a": a, "beta": beta, "gamma": gamma, "sd_e": sd_e,
                     "lam": lam, "true_adj_slope_std": true_slope,
                     "corrected_slope": corr_slope, "abs_disc": disc})
    # empirical brute force
    rng2 = np.random.default_rng(2026102703)
    emp = []
    n = 1_000_000
    for (a, beta, gamma, sd_e, lam) in [(0.6, 0.4, 0.8, 1.0, 0.64),
                                        (-0.5, -0.3, 0.5, 0.8, 0.8),
                                        (0.3, 0.0, 1.0, 1.2, 0.5)]:
        z = rng2.standard_normal(n)
        ex = rng2.standard_normal(n)
        x = a * z + np.sqrt(1 - a * a) * ex
        y = beta * x + gamma * z + sd_e * rng2.standard_normal(n)
        u = np.sqrt((1 - lam) / lam) * rng2.standard_normal(n)
        zs = z + u
        def std(v):
            return (v - v.mean()) / v.std()
        Xo = np.column_stack([std(x), std(zs), std(y)])
        S_obs = np.cov(Xo, rowvar=False)
        corr_slope = corrected_slope(S_obs, lam)
        Xt = np.column_stack([std(x), std(z), std(y)])
        S_tr = np.cov(Xt, rowvar=False)
        true_slope = adj_slope_from_cov(S_tr)
        emp.append({"a": a, "beta": beta, "gamma": gamma, "sd_e": sd_e,
                    "lam": lam, "true_adj_slope_sample": true_slope,
                    "corrected_slope_sample": corr_slope,
                    "abs_disc": abs(true_slope - corr_slope)})
    out = {"spec": "upgrade2_preregistration.md Section 2",
           "population_settings": 500,
           "population_max_abs_discrepancy": float(max_disc),
           "population_rows_first5": rows[:5],
           "empirical_n": n,
           "empirical": emp,
           "empirical_max_abs_discrepancy": float(max(e["abs_disc"] for e in emp))}
    (HERE / "algebra_results.json").write_text(json.dumps(out, indent=2))
    print("population max |disc| =", max_disc)
    print("empirical max |disc| =", out["empirical_max_abs_discrepancy"])


if __name__ == "__main__":
    main()
