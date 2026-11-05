"""Probe 4 -- numerical demonstration of the sequential results of Section 4
(Theorem: sequential alpha-budget; Corollary: lifetime alpha-spending;
Proposition: predictability is necessary).

Design (all seeds fixed via zlib.crc32 of a string label -- deterministic,
no salted hash()):
  (1) A predictable, history-adaptive policy: at each step the battery
      (m = 3, disjoint intervals on a uniform pivot, endpoints = cumulative
      sums of the predictable levels) is chosen from H_{t-1} only:
        - if step t-1 validated, "cool down": B_t = 0.02,
          alphas = (0.010, 0.006, 0.004);
        - otherwise B_t = 0.05, with the allocation rotating with the
          parity of the number of validations so far:
          even -> (0.025, 0.015, 0.010), odd -> (0.010, 0.025, 0.015).
      Per-step budget B_t <= 0.05 throughout; the theorem bounds the
      long-run validation rate by limsup of the average budget <= 0.05,
      and the attaining construction makes the conditional validation
      probability exactly B_t, so the rate should track the mean budget.
  (2) Lifetime schedule B_t = A * (6/pi^2) / t^2 with A = 0.5
      (sum_{t>=1} B_t = A exactly). R independent runs of length N2 with
      the attaining construction (validation iff U_t < B_t): total
      validations per run must be a small finite number, mean <= A,
      and the cumulative budget over the run must be ~= A.
  (3) Sharpness: predictable schedule cycling B_t through
      (0.02, 0.04, 0.05, 0.03) by t mod 4 (mean budget 0.035), battery
      m = 4 with weights (0.4, 0.3, 0.2, 0.1) and disjoint intervals
      whose endpoints are the cumulative sums of alpha_{t,j}.
      Empirical rate must match the mean budget 0.035.
  (4) Counterexample (predictability dropped): pivot U ~ Uniform[0,1].
      Interval family I_u = [u, u+alpha] (mod 1), alpha = 0.05; every
      FIXED u is a test of size alpha (control run, u = 0.3). Choosing
      u = U after seeing the candidate validates with probability 1.
      Singleton family {x}: size 0 under the atomless pivot law; fixed
      x = 0.3 never validates, choosing x = U validates always.
"""
import zlib

import numpy as np


def rng(label):
    return np.random.default_rng(zlib.crc32(label.encode("ascii")))


print("PROBE 4 -- sequential alpha-budget, lifetime spending, sharpness,")
print("           and the predictability counterexample (Paper 2, Section 4)")
print("seeds: zlib.crc32 of fixed string labels (deterministic)")

# ---------------------------------------------------------------- (1)
print("\n(1) Predictable history-adaptive policy, per-step cap B_t <= 0.05")
N = 200_000
r1 = rng("paper02-probe4-adaptive")
U = r1.random(N)
V = 0
budget_sum = 0.0
max_B = 0.0
prev_valid = False
for t in range(N):
    if prev_valid:
        alphas = (0.010, 0.006, 0.004)
    elif V % 2 == 0:
        alphas = (0.025, 0.015, 0.010)
    else:
        alphas = (0.010, 0.025, 0.015)
    B_t = sum(alphas)
    budget_sum += B_t
    max_B = max(max_B, B_t)
    # disjoint intervals with cumulative-sum endpoints on the pivot
    lo = 0.0
    valid = False
    for a in alphas:
        if lo <= U[t] < lo + a:
            valid = True
            break
        lo += a
    prev_valid = valid
    V += int(valid)
print(f"    steps N                 = {N}")
print(f"    max per-step budget B_t = {max_B:.4f}")
print(f"    mean budget (1/N)sum B_t= {budget_sum / N:.4f}")
print(f"    cumulative budget       = {budget_sum:.2f}")
print(f"    validations V_N         = {V}")
print(f"    empirical rate V_N/N    = {V / N:.4f}   (cap 0.0500 respected: "
      f"{V / N <= 0.05})")

# ---------------------------------------------------------------- (2)
print("\n(2) Lifetime alpha-spending: B_t = A*(6/pi^2)/t^2, A = 0.5")
A = 0.5
c = 6.0 / np.pi**2
N2 = 50_000
tt = np.arange(1, N2 + 1, dtype=float)
B = A * c / tt**2
cum_budget = B.sum()
full_budget = A * c * (np.pi**2 / 6.0)  # sum over all t >= 1
R = 2000
r2 = rng("paper02-probe4-lifetime")
totals = np.empty(R, dtype=int)
for s in range(0, R, 100):  # chunked to bound memory
    U2 = r2.random((100, N2))
    totals[s:s + 100] = (U2 < B).sum(axis=1)
print(f"    runs R x length         = {R} x {N2}")
print(f"    sum_t B_t over run      = {cum_budget:.4f}  (lifetime total "
      f"over all t: {full_budget:.4f} = A)")
print(f"    total validations: mean = {totals.mean():.4f}, "
      f"min = {totals.min()}, max = {totals.max()}")
print(f"    predicted E[total]      = {cum_budget:.4f} <= A = {A:.4f} "
      f"(theorem); sample mean estimates this quantity")
print(f"    fraction of runs V>=1   = {(totals >= 1).mean():.4f}  "
      f"(bound Pr[V_inf>=1] <= A = {A:.4f})")
print(f"    long-run rate, run 0    = {totals[0] / N2:.6f}  "
      f"(V_N finite => rate -> 0)")

# ---------------------------------------------------------------- (3)
print("\n(3) Sharpness: attaining construction, predictable B_t schedule")
N3 = 400_000
r3 = rng("paper02-probe4-sharpness")
U3 = r3.random(N3)
cycle = np.array([0.02, 0.04, 0.05, 0.03])
weights = np.array([0.4, 0.3, 0.2, 0.1])
B3 = cycle[np.arange(N3) % 4]
flags3 = np.zeros(N3, dtype=bool)
for k in range(4):
    idx = np.arange(N3) % 4 == k
    bounds = np.concatenate([[0.0], np.cumsum(cycle[k] * weights)])
    u = U3[idx]
    f = np.zeros(idx.sum(), dtype=bool)
    for j in range(4):
        f |= (u >= bounds[j]) & (u < bounds[j + 1])
    flags3[idx] = f
print(f"    steps N                 = {N3}")
print(f"    mean budget (1/N)sum B_t= {B3.mean():.4f}")
print(f"    empirical rate          = {flags3.mean():.4f}")
print(f"    |rate - mean budget|    = {abs(flags3.mean() - B3.mean()):.4f}")

# ---------------------------------------------------------------- (4)
print("\n(4) Counterexample: candidate-dependent test construction")
N4 = 200_000
r4 = rng("paper02-probe4-counterexample")
U4 = r4.random(N4)
alpha = 0.05
# fixed u chosen ex ante (control): size alpha
fixed = ((U4 - 0.3) % 1.0) <= alpha
print(f"    interval test, FIXED u=0.3, nominal alpha = {alpha:.4f}: "
      f"rate = {fixed.mean():.4f}")
# u chosen after seeing the candidate: u = U validates always
chosen = ((U4 - U4) % 1.0) <= alpha
print(f"    interval test, u = U_t chosen after candidate, nominal "
      f"alpha = {alpha:.4f}: rate = {chosen.mean():.4f} "
      f"(count {chosen.sum()}/{N4})")
# singleton family {x}, size 0 under atomless pivot
fixed_single = U4 == 0.3
print(f"    singleton test, FIXED x=0.3, nominal size = 0.0000: "
      f"rate = {fixed_single.mean():.4f}")
chosen_single = U4 == U4
print(f"    singleton test, x = U_t chosen after candidate, nominal "
      f"size = 0.0000: rate = {chosen_single.mean():.4f} "
      f"(count {chosen_single.sum()}/{N4})")
print("\nPROBE4 DONE")
