"""Forecasting cardiovascular risk from the resting ECG: horizon durability,
multi-modal redundancy, and knowledge-graph-guided testing (CARDIO).

Original implementation of the core computational ideas in
"Preliminary investigation: forecasting cardiovascular risk from the resting
ECG" (Dr. Ranadhir Ghosh, shared with Dr. Debabrata Mukhopadhyay, Mayo Clinic,
Sep 2026). Implemented here on SYNTHETIC data (no cohort access): a
constant-effect proportional-hazards world with redundant ECG views, used to
demonstrate the paper's measurement protocol --
  (a) the Grambsch-Therneau test for time-varying coefficients,
  (b) a constant-effect null simulation that calibrates horizon trends,
  (c) an interaction-information screen for modality synergy that gates
      fusion-architecture investment (concatenation vs gated fusion),
  (d) a typed knowledge graph whose structural tensions flag
      invariance-vs-trend contradictions,
plus a 9-layer gated pipeline in which a failed gate blocks the run.

Numpy only (scipy for chi-square tails); vectorized risk sets, chunked
pairwise statistics.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

__all__ = [
    "FEATURE_NAMES",
    "simulate_cohort",
    "partial_loglik",
    "fit_cox",
    "schoenfeld",
    "grambsch_therneau",
    "km_survival",
    "km_eval",
    "breslow",
    "h0_eval",
    "predict_surv",
    "uno_concordance",
    "td_auc",
    "ipcw_brier",
    "ipa",
    "landmark_cohort",
    "interaction_information",
    "ClaimGraph",
    "GatedPipeline",
    "fit_gated_fusion",
]

FEATURE_NAMES = ["age", "sex", "ecg1", "ecg2", "annot"]

# True data-generating coefficients (constant in time): eta =
#   B_AGE*(age-55) + B_SEX*sex + B_L*L
B_AGE, B_SEX, B_L = 0.06, 0.4, 0.9
LAM0 = 0.04  # exponential baseline rate -> ~25% events with U(0.5, 8.5) censoring


def simulate_cohort(
    n: int = 4000,
    seed: int = 0,
    het_k: float = 0.0,
    het_thresh: float = 0.75,
    lam0: float = LAM0,
) -> dict:
    """Synthetic constant-effect proportional-hazards cohort.

    Covariates: age ~ N(55, 12), sex ~ Bernoulli(0.5), latent risk L ~ N(0, 1).
    Redundant ECG views of the same latent signal (all functions of L):
      ecg1  = L + 0.5*N(0,1)        (learned-embedding proxy)
      ecg2  = 0.8*L + 0.6*N(0,1)    (second embedding view)
      annot = 0.5*L + N(0,1)        (expert-annotation proxy)
    True linear predictor eta = 0.06*(age-55) + 0.4*sex + 0.9*L with
    CONSTANT coefficients. Event time T = -log(U)/(lam0*exp(eta))
    (exponential baseline); censoring C ~ Uniform(0.5, 8.5) years.
    Returns dict with Y = min(T, C), delta, X (n x 5), and blocks.

    het_k > 0 enables the miniature's Result-3 mechanism: patients with
    L < het_thresh get ECG-view noise multiplied by (1 + het_k), i.e.
    risk-correlated measurement noise. The true coefficients stay constant,
    so the Grambsch-Therneau test remains blind to it, but the ECG views
    carry less ranking information for late (low-risk) events, which bends
    the incremental concordance down with the horizon -- the paper's
    unexplained drift, reproduced with a concrete candidate mechanism.
    het_k = 0 (default) is the pure constant-effect world used for the
    null simulation.
    """
    rng = np.random.default_rng(seed)
    age = rng.normal(55.0, 12.0, size=n)
    sex = rng.binomial(1, 0.5, size=n).astype(float)
    L = rng.normal(0.0, 1.0, size=n)
    mult = np.where(L < het_thresh, 1.0 + het_k, 1.0)
    ecg1 = L + 0.5 * rng.normal(0.0, 1.0, size=n) * mult
    ecg2 = 0.8 * L + 0.6 * rng.normal(0.0, 1.0, size=n) * mult
    annot = 0.5 * L + rng.normal(0.0, 1.0, size=n) * mult
    eta = B_AGE * (age - 55.0) + B_SEX * sex + B_L * L
    T = -np.log(rng.random(size=n)) / (lam0 * np.exp(eta))
    C = rng.uniform(0.5, 8.5, size=n)
    Y = np.minimum(T, C)
    delta = (T <= C).astype(float)
    X = np.column_stack([age - 55.0, sex, ecg1, ecg2, annot])
    return {
        "X": X, "Y": Y, "delta": delta, "names": list(FEATURE_NAMES),
        "blocks": {
            "demo": X[:, :2],          # age, sex
            "ecg": X[:, 2:4],          # ecg1, ecg2
            "annot": X[:, 4:5],        # expert annotation proxy
        },
        "n": n, "seed": seed, "het_k": het_k,
        "event_rate": float(delta.mean()),
    }


def _descending(Y: np.ndarray):
    return np.argsort(-Y, kind="stable")


def partial_loglik(beta: np.ndarray, X: np.ndarray, Y: np.ndarray,
                   delta: np.ndarray) -> float:
    """Breslow partial log-likelihood at beta (risk sets via cumsum)."""
    beta = np.asarray(beta, dtype=float)
    o = _descending(Y)
    Xs, ds = X[o], delta[o]
    eta = Xs @ beta
    w = np.exp(eta - eta.max())  # scale-free: only ratios enter
    S0 = np.cumsum(w)
    ll = float(np.sum((eta - np.log(S0))[ds == 1]))
    return ll


def fit_cox(X: np.ndarray, Y: np.ndarray, delta: np.ndarray,
            max_iter: int = 50, tol: float = 1e-9) -> dict:
    """Cox PH via Newton-Raphson on the Breslow partial likelihood.

    Observations sorted by descending Y; risk-set sums S0/S1/S2 via cumsum.
    Returns dict(beta, loglik, n_iter, converged).
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    o = _descending(Y)
    Xs, ds = X[o], np.asarray(delta)[o]
    beta = np.zeros(p)
    converged, n_iter = False, 0
    for it in range(max_iter):
        eta = Xs @ beta
        w = np.exp(eta - eta.max())
        S0 = np.cumsum(w)
        S1 = np.cumsum(w[:, None] * Xs, axis=0)
        xbar = S1 / S0[:, None]
        S2 = np.cumsum(w[:, None, None] * Xs[:, :, None] * Xs[:, None, :],
                       axis=0)
        V = S2 / S0[:, None, None] - xbar[:, :, None] * xbar[:, None, :]
        ev = ds == 1
        grad = np.sum(Xs[ev] - xbar[ev], axis=0)
        info = np.sum(V[ev], axis=0) + 1e-10 * np.eye(p)
        step = np.linalg.solve(info, grad)
        beta = beta + step
        n_iter = it + 1
        if float(np.max(np.abs(step))) < tol:
            converged = True
            break
    return {"beta": beta, "loglik": partial_loglik(beta, X, Y, delta),
            "n_iter": n_iter, "converged": converged}


def schoenfeld(beta: np.ndarray, X: np.ndarray, Y: np.ndarray,
               delta: np.ndarray) -> dict:
    """Schoenfeld residuals (per event: x_i - xbar_i) and scaled versions.

    Scaled residuals = d * Info^{-1} @ r, where d = #events. Returns dict
    with resid (d x p), scaled (d x p), info (p x p), times (d,).
    """
    X = np.asarray(X, dtype=float)
    beta = np.asarray(beta, dtype=float)
    o = _descending(Y)
    Xs, Ys, ds = X[o], np.asarray(Y)[o], np.asarray(delta)[o]
    eta = Xs @ beta
    w = np.exp(eta - eta.max())
    S0 = np.cumsum(w)
    xbar = np.cumsum(w[:, None] * Xs, axis=0) / S0[:, None]
    S2 = np.cumsum(w[:, None, None] * Xs[:, :, None] * Xs[:, None, :], axis=0)
    V = S2 / S0[:, None, None] - xbar[:, :, None] * xbar[:, None, :]
    ev = ds == 1
    resid = Xs[ev] - xbar[ev]
    info = np.sum(V[ev], axis=0) + 1e-10 * np.eye(X.shape[1])
    d = int(ev.sum())
    scaled = d * np.linalg.solve(info, resid.T).T
    return {"resid": resid, "scaled": scaled, "info": info,
            "times": Ys[ev], "n_events": d}


def grambsch_therneau(beta: np.ndarray, X: np.ndarray, Y: np.ndarray,
                      delta: np.ndarray) -> dict:
    """Grambsch-Therneau test of proportional hazards.

    Tests whether scaled Schoenfeld residuals trend with event time g:
      chi2_k = [sum((g-gbar)*r_k)]^2 / (d * V_kk * sum((g-gbar)^2))
    with V = Info^{-1}, plus a global chi2(p) test. Under constant effects
    all p-values should be well above 0.05. Returns dict with per-covariate
    p-values, chi2 statistics, and the global test.
    """
    sch = schoenfeld(beta, X, Y, delta)
    r, g = sch["scaled"], sch["times"]
    d = sch["n_events"]
    p_ = r.shape[1]
    gc = g - g.mean()
    Vbar = np.linalg.inv(sch["info"])
    Vkk = np.diag(Vbar)
    U = gc @ r  # (p,)
    gs = float(np.sum(gc ** 2))
    chi2s = U ** 2 / (d * Vkk * gs)
    pvals = chi2.sf(chi2s, 1)
    gstat = float(U @ np.linalg.solve(d * Vbar, U) / gs)
    return {"p_values": pvals, "chi2": chi2s,
            "global_chi2": gstat,
            "global_p": float(chi2.sf(gstat, p_)),
            "n_events": d}


def km_survival(T: np.ndarray, delta: np.ndarray):
    """Kaplan-Meier step function. Returns (times, surv) with times[0]=0."""
    o = np.argsort(T, kind="stable")
    Ts, ds = np.asarray(T)[o], np.asarray(delta)[o]
    n = len(T)
    at_risk = n - np.arange(n)
    times, ss = [0.0], [1.0]
    s = 1.0
    for k in range(n):
        if ds[k] == 1:
            s *= 1.0 - 1.0 / at_risk[k]
            times.append(float(Ts[k]))
            ss.append(s)
    return np.array(times), np.array(ss)


def km_eval(times: np.ndarray, surv: np.ndarray, t) -> np.ndarray:
    """Evaluate a KM step function at t (right-continuous)."""
    idx = np.searchsorted(times, np.atleast_1d(t), side="right") - 1
    return surv[np.clip(idx, 0, len(surv) - 1)]


def breslow(beta: np.ndarray, X: np.ndarray, Y: np.ndarray,
            delta: np.ndarray):
    """Breslow cumulative baseline hazard steps. Returns (times, H0)."""
    X = np.asarray(X, dtype=float)
    beta = np.asarray(beta, dtype=float)
    o = _descending(Y)
    Xs, Ys, ds = X[o], np.asarray(Y)[o], np.asarray(delta)[o]
    eta = Xs @ beta
    w = np.exp(eta - eta.max())
    S0 = np.cumsum(w)
    ev = ds == 1
    # undo the stability shift: true S0 = S0*exp(max), dH = 1/S0_true
    dH = np.where(ev, np.exp(-eta.max()) / S0, 0.0)
    oa = np.argsort(Ys, kind="stable")
    return Ys[oa], np.cumsum(dH[oa])


def h0_eval(times: np.ndarray, H: np.ndarray, tau: float) -> float:
    """Cumulative baseline hazard at tau."""
    idx = np.searchsorted(times, tau, side="right") - 1
    return float(H[np.clip(idx, 0, len(H) - 1)])


def predict_surv(eta: np.ndarray, H0_tau: float) -> np.ndarray:
    """Predicted P(T > tau) = exp(-H0(tau) * exp(eta))."""
    return np.exp(-H0_tau * np.exp(np.asarray(eta, dtype=float)))


def uno_concordance(eta: np.ndarray, Y: np.ndarray, delta: np.ndarray,
                    tau: float, G) -> float:
    """Uno's censoring-weighted concordance truncated at tau.

    Comparable pairs: i with delta_i = 1, Y_i <= tau, Y_i < Y_j;
    pair weight 1/G(Y_i) where G is the censoring KM (callable).
    Vectorized in chunks over events (no O(n^2) blowup).
    """
    eta = np.asarray(eta, dtype=float)
    Y = np.asarray(Y, dtype=float)
    delta = np.asarray(delta, dtype=float)
    ev = (delta == 1) & (Y <= tau)
    ei = np.flatnonzero(ev)
    if len(ei) == 0:
        return float("nan")
    num = den = 0.0
    for a in range(0, len(ei), 256):
        ii = ei[a:a + 256]
        comp = Y[ii, None] < Y[None, :]
        conc = eta[ii, None] > eta[None, :]
        tie = eta[ii, None] == eta[None, :]
        w = 1.0 / np.maximum(G(Y[ii]), 1e-6)
        num += float(np.sum(w[:, None] * comp * (conc + 0.5 * tie)))
        den += float(np.sum(w[:, None] * comp))
    return num / den if den > 0 else float("nan")


def td_auc(eta: np.ndarray, Y: np.ndarray, delta: np.ndarray,
           tau: float, G) -> float:
    """Cumulative/dynamic time-dependent AUC at tau.

    Cases: Y <= tau & delta = 1, weight 1/G(Y); controls: Y > tau,
    weight 1/G(tau). Chunked over cases.
    """
    eta = np.asarray(eta, dtype=float)
    Y = np.asarray(Y, dtype=float)
    delta = np.asarray(delta, dtype=float)
    ci = np.flatnonzero((delta == 1) & (Y <= tau))
    ni = np.flatnonzero(Y > tau)
    if len(ci) == 0 or len(ni) == 0:
        return float("nan")
    wc = 1.0 / np.maximum(G(Y[ci]), 1e-6)
    wn = 1.0 / max(float(G(np.array([tau]))[0]), 1e-6)
    num = 0.0
    for a in range(0, len(ci), 256):
        cc = ci[a:a + 256]
        hit = eta[cc, None] > eta[None, ni]
        num += float(np.sum(wc[a:a + 256, None] * hit))
    den = float(wc.sum() * wn * len(ni))
    return num / den if den > 0 else float("nan")


def ipcw_brier(eta: np.ndarray, H0_tau: float, Y: np.ndarray,
               delta: np.ndarray, tau: float, G,
               S_ref_tau: float) -> tuple:
    """IPCW Brier score at tau and the reference (KM) Brier.

    BS = mean over i of I(Y_i > tau)*(1 - S_i)^2/G(tau)
                       + I(Y_i <= tau, delta_i = 1)*S_i^2/G(Y_i),
    with S_i the model's P(T > tau); the reference uses the KM survival.
    Returns (bs_model, bs_ref).
    """
    S = predict_surv(eta, H0_tau)
    Y = np.asarray(Y, dtype=float)
    delta = np.asarray(delta, dtype=float)
    n = len(Y)
    m1 = Y > tau
    m2 = (Y <= tau) & (delta == 1)
    Gt = max(float(G(np.array([tau]))[0]), 1e-6)
    bs = (np.sum((1 - S[m1]) ** 2) / Gt
          + np.sum(S[m2] ** 2 / np.maximum(G(Y[m2]), 1e-6))) / n
    bs_ref = (np.sum(m1) * (1 - S_ref_tau) ** 2 / Gt
              + np.sum(m2 * S_ref_tau ** 2 / np.maximum(G(Y), 1e-6))) / n
    return float(bs), float(bs_ref)


def ipa(bs_model: float, bs_ref: float) -> float:
    """Index of prediction accuracy: 1 - BS(model)/BS(KM null)."""
    return 1.0 - bs_model / bs_ref if bs_ref > 0 else float("nan")


def landmark_cohort(Y: np.ndarray, delta: np.ndarray,
                    tau_max: float = 7.5) -> np.ndarray:
    """Fixed-cohort landmark mask: outcome known at tau_max.

    mask = (Y > tau_max) | (delta == 1). The same population is carried
    across all horizons so the curve reflects the horizon alone.
    """
    Y = np.asarray(Y, dtype=float)
    delta = np.asarray(delta, dtype=float)
    return (Y > tau_max) | (delta == 1)


def interaction_information(XA: np.ndarray, XB: np.ndarray, Y: np.ndarray,
                            delta: np.ndarray) -> dict:
    """Interaction information II(A;B) = I(A,B) - I(A) - I(B).

    I(S) = (l_S - l_null)/2 with l the Cox partial log-likelihood and
    l_null the beta=0 value. Negative II = redundant views (less than the
    sum of parts); positive = synergy. Returns dict(II, I_A, I_B, I_AB).
    """
    l_null = partial_loglik(np.zeros(XA.shape[1] + XB.shape[1]),
                            np.column_stack([XA, XB]), Y, delta)
    fitA = fit_cox(XA, Y, delta)
    fitB = fit_cox(XB, Y, delta)
    fitAB = fit_cox(np.column_stack([XA, XB]), Y, delta)
    IA = (fitA["loglik"] - l_null) / 2.0
    IB = (fitB["loglik"] - l_null) / 2.0
    IAB = (fitAB["loglik"] - l_null) / 2.0
    return {"II": IAB - IA - IB, "I_A": IA, "I_B": IB, "I_AB": IAB}


class ClaimGraph:
    """Typed knowledge graph: claims tagged theorem / measured / assumption.

    Each claim asserts a relation of a variable: 'invariant' (to a stated
    axis), 'trend_up', or 'trend_down'. detect_tensions() applies the
    paper's structural rule: an edge asserting invariance of a variable
    alongside an edge asserting a trend of the SAME variable is a tension.
    """

    KINDS = ("theorem", "measured", "assumption")
    RELATIONS = ("invariant", "trend_up", "trend_down")

    def __init__(self):
        self.claims = []

    def add_claim(self, kind: str, variable: str, relation: str,
                  statement: str = "", axis: str = "") -> int:
        if kind not in self.KINDS:
            raise ValueError(f"kind must be one of {self.KINDS}")
        if relation not in self.RELATIONS:
            raise ValueError(f"relation must be one of {self.RELATIONS}")
        self.claims.append({"kind": kind, "variable": variable,
                            "relation": relation, "statement": statement,
                            "axis": axis})
        return len(self.claims) - 1

    def detect_tensions(self):
        """Return [(invariant_claim, trend_claim)] pairs on the same variable."""
        tensions = []
        for i, ci in enumerate(self.claims):
            for j, cj in enumerate(self.claims):
                if i >= j:
                    continue
                if ci["variable"] != cj["variable"]:
                    continue
                ri, rj = ci["relation"], cj["relation"]
                if ri == "invariant" and rj in ("trend_up", "trend_down"):
                    tensions.append((ci, cj))
                elif rj == "invariant" and ri in ("trend_up", "trend_down"):
                    tensions.append((cj, ci))
        return tensions

    def summary(self) -> dict:
        return {"n_claims": len(self.claims),
                "n_tensions": len(self.detect_tensions())}


class GatedPipeline:
    """Layered pipeline where a failed gate BLOCKS the run.

    layers: list of dicts with 'name', 'run_fn' (ctx -> result) and
    'gate_fn' (ctx, result -> (ok: bool, reason: str)). run() executes in
    order and STOPS at the first gate failure; the ledger records
    (layer, status) with status in {'pass', 'blocked', 'failed'}.
    A check that does not stop the run is a comment -- here it stops it.
    """

    def __init__(self, layers):
        self.layers = list(layers)
        self.ledger = []
        self.context = {}

    def run(self) -> dict:
        self.ledger = []
        self.context = {}
        for layer in self.layers:
            name = layer["name"]
            try:
                result = layer["run_fn"](self.context)
                self.context[name] = result
            except Exception as exc:  # noqa: BLE001
                self.ledger.append((name, f"failed: {exc}"))
                return {"completed": False, "blocked_at": name,
                        "ledger": list(self.ledger)}
            ok, reason = layer["gate_fn"](self.context, result)
            if ok:
                self.ledger.append((name, "pass"))
            else:
                self.ledger.append((name, f"blocked: {reason}"))
                return {"completed": False, "blocked_at": name,
                        "ledger": list(self.ledger)}
        return {"completed": True, "blocked_at": None,
                "ledger": list(self.ledger)}


def fit_gated_fusion(blocks: dict, Y: np.ndarray, delta: np.ndarray,
                     seed: int = 0, steps: int = 200, lr: float = 0.05,
                     hidden: int = 4) -> dict:
    """Tiny gated-fusion network trained on the Cox partial likelihood.

    Per-modality tanh encoder h_m = tanh(x_m @ W_m + b_m); softmax attention
    a = softmax(v^T h_m + c) over modalities; risk score
    eta = sum_m a_m * (h_m @ w_m + b_m). Trained with full-batch Adam on
    the negative Breslow partial likelihood (+ tiny L2). Numpy only.

    blocks: dict of modality name -> (n x p_m) standardized features.
    Returns dict(eta, weights, n_params, final_loss).
    """
    rng = np.random.default_rng(seed)
    names = list(blocks.keys())
    Xs = {k: np.asarray(v, dtype=float) for k, v in blocks.items()}
    n = len(Y)
    o = _descending(Y)
    Xo = {k: v[o] for k, v in Xs.items()}
    do = np.asarray(delta)[o] == 1

    params = {}
    for k, v in Xo.items():
        p = v.shape[1]
        params[k] = {
            "W": rng.normal(0, 0.5, size=(p, hidden)),
            "b": np.zeros(hidden),
            "w": rng.normal(0, 0.5, size=hidden),
            "c": 0.0,
        }
    vv = rng.normal(0, 0.5, size=hidden)

    def forward(P, V):
        H, S = {}, {}
        logits = []
        for k in names:
            H[k] = np.tanh(Xo[k] @ P[k]["W"] + P[k]["b"])
            S[k] = H[k] @ P[k]["w"] + P[k]["c"]
            logits.append(H[k] @ V)
        L = np.column_stack(logits)
        L = L - L.max(axis=1, keepdims=True)
        A = np.exp(L)
        A = A / A.sum(axis=1, keepdims=True)
        eta_o = np.sum(A * np.column_stack([S[k] for k in names]), axis=1)
        cache = {"H": H, "S": S, "A": A, "L": L, "eta_o": eta_o}
        return eta_o, cache

    def loss_and_grads(P, V):
        eta_o, cache = forward(P, V)
        w = np.exp(eta_o - eta_o.max())
        S0 = np.cumsum(w)
        nll = -float(np.sum((eta_o - np.log(S0))[do]))
        reg = 1e-4 * (float(V @ V) + sum(float((P[k]["W"] ** 2).sum())
                                        for k in names))
        # d nll / d eta_o
        d_eta = np.zeros(n)
        d_eta[do] = -1.0
        # d/d eta_o of -sum_{ev}(eta - log S0): -[1_ev - w_i * cumsum(1_ev/S0)_i]
        inv = np.where(do, 1.0 / S0, 0.0)
        cs = np.cumsum(inv)
        d_eta = d_eta + w * cs
        grads = {k: {"W": np.zeros_like(P[k]["W"]),
                     "b": np.zeros(hidden),
                     "w": np.zeros(hidden), "c": 0.0}
                 for k in names}
        gV = 1e-4 * 2 * V
        A, H, S = cache["A"], cache["H"], cache["S"]
        Sm = np.column_stack([S[k] for k in names])
        # d eta / d A and d eta / d S
        dA = d_eta[:, None] * Sm
        dS = d_eta[:, None] * A
        # softmax backward: dL = A * (dA - sum(dA*A, axis=1, keepdims))
        dL = A * (dA - np.sum(dA * A, axis=1, keepdims=True))
        for j, k in enumerate(names):
            # attention logit l_k = H_k @ V
            gV = gV + H[k].T @ dL[:, j]
            dH_v = np.outer(dL[:, j], V)  # via attention path
            # score path
            dH_s = np.outer(dS[:, j], P[k]["w"])
            dH = (dH_v + dH_s) * (1 - H[k] ** 2)  # tanh'
            grads[k]["W"] = Xo[k].T @ dH + 1e-4 * 2 * P[k]["W"]
            grads[k]["b"] = dH.sum(axis=0)
            grads[k]["w"] = H[k].T @ dS[:, j]
            grads[k]["c"] = float(dS[:, j].sum())
        return nll + reg, grads, gV

    # Adam
    mP = {k: {q: np.zeros_like(v) for q, v in params[k].items()}
          for k in names}
    vP = {k: {q: np.zeros_like(v) for q, v in params[k].items()}
          for k in names}
    mV = np.zeros_like(vv)
    vV = np.zeros_like(vv)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        nll, grads, gV = loss_and_grads(params, vv)
        for k in names:
            for q in params[k]:
                mP[k][q] = b1 * mP[k][q] + (1 - b1) * grads[k][q]
                vP[k][q] = b2 * vP[k][q] + (1 - b2) * grads[k][q] ** 2
                mhat = mP[k][q] / (1 - b1 ** t)
                vhat = vP[k][q] / (1 - b2 ** t)
                params[k][q] -= lr * mhat / (np.sqrt(vhat) + eps)
        mV = b1 * mV + (1 - b1) * gV
        vV = b2 * vV + (1 - b2) * gV ** 2
        vv -= lr * (mV / (1 - b1 ** t)) / (np.sqrt(vV / (1 - b2 ** t)) + eps)

    eta_o, cache = forward(params, vv)
    eta = np.empty(n)
    eta[o] = eta_o
    n_params = sum(v.size for k in names for v in params[k].values()) + vv.size
    return {"eta": eta, "attention": cache["A"][np.argsort(o)],
            "n_params": n_params, "final_nll": float(nll),
            "modalities": names}
