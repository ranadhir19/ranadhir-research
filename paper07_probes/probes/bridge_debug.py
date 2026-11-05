"""Debug: per-model log marginal likelihoods — bridge vs quadrature vs
importance sampling — on two edges (null_demo, itsm_p1_drives_churn),
plus a conjugate toy problem with a known marginal likelihood."""
import numpy as np
from scipy import stats, special
import importlib.util, sys
spec = importlib.util.spec_from_file_location(
    "brm", "/home/hatch/workspace/goals/10-journal-papers-submitted-to-good-journals/hidden_files/paper07_work/probes/bridge_rhat_multiseed.py")
brm = importlib.util.module_from_spec(spec)
sys.modules["brm"] = brm
spec.loader.exec_module(brm) if False else None
# avoid running main(): the module only defines functions + main guard, safe to exec
spec.loader.exec_module(brm)

rng = np.random.RandomState(0)

def is_logml(samples_phi, logq, rng, n_prop=200000):
    """Importance-sampling logml with the fitted MVN proposal (independent check)."""
    mu = samples_phi.mean(axis=0); cov = np.cov(samples_phi.T)
    prop = rng.multivariate_normal(mu, cov, size=n_prop)
    lq = np.array([logq(s) for s in prop])
    lg = stats.multivariate_normal.logpdf(prop, mu, cov)
    return float(special.logsumexp(lq - lg) - np.log(n_prop))

for idx in (0, 18):
    slug, bs, n, xlo, xhi, sig, d = brm.BENCH[idx]
    x, y = brm.gen_data(bs, n, xlo, xhi, sig, seed=idx)
    xc = x - x.mean()
    sb = 5.0
    # M1
    s1, _ = brm.run_chain(xc, y, sb, idx, "m1")
    phi1 = np.column_stack([s1[:, 0], s1[:, 1], np.log(s1[:, 2])])
    lq1 = lambda phi: brm.log_post_m1((phi[0], phi[1], np.exp(phi[2])), xc, y, sb) + phi[2]
    # M0
    s0, _ = brm.run_chain(xc, y, sb, idx + 55, "m0")
    phi0 = np.column_stack([s0[:, 0], np.log(s0[:, 1])])
    lq0 = lambda phi: brm.log_post_m0((phi[0], np.exp(phi[1])), xc, y, sb) + phi[1]
    b1 = brm.bridge_logml(phi1, lq1, np.random.RandomState(1))
    b0 = brm.bridge_logml(phi0, lq0, np.random.RandomState(2))
    q1 = brm.quad_logml(xc, y, sb, "m1"); q0 = brm.quad_logml(xc, y, sb, "m0")
    i1 = is_logml(phi1, lq1, np.random.RandomState(3))
    i0 = is_logml(phi0, lq0, np.random.RandomState(4))
    print(f"{slug}: ")
    print(f"  logml1  bridge={b1:.3f} quad={q1:.3f} IS={i1:.3f}")
    print(f"  logml0  bridge={b0:.3f} quad={q0:.3f} IS={i0:.3f}")
    print(f"  logBF10 bridge={b1-b0:.3f} quad={q1-q0:.3f} IS={i1-i0:.3f}")

# Toy: y_i ~ N(mu, 1), mu ~ N(0, 3^2); exact logml known in closed form.
n = 50; y = np.random.RandomState(7).normal(0.7, 1.0, n)
# exact: y ~ N(0, I + 9 11')
S = np.eye(n) + 9.0 * np.ones((n, n))
exact = stats.multivariate_normal.logpdf(y, np.zeros(n), S)
samp = np.random.RandomState(8).multivariate_normal(
    [ (n * y.mean() / (n + 1/9.0)) ], [[1.0 / (n + 1/9.0)]], size=4000)
def lq_toy(phi):
    mu = phi[0]
    return float(np.sum(stats.norm.logpdf(y, mu, 1.0)) + stats.norm.logpdf(mu, 0, 3.0))
bt = brm.bridge_logml(samp, lq_toy, np.random.RandomState(9))
it = is_logml(samp, lq_toy, np.random.RandomState(10))
print(f"toy: exact={exact:.3f} bridge={bt:.3f} IS={it:.3f}")
