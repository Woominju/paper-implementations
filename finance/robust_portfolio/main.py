"""
Robust portfolio optimization with a worst-case approach
(Kim, Kim & Fabozzi, JOTA 2014, survey), illustrated with an ellipsoidal
uncertainty set for expected returns.

Classical mean-variance optimization plugs in estimated means mu_hat, which
are noisy, and tends to put large weights on assets whose returns were
overestimated. The robust version maximizes the worst case over
    U = { mu : (mu - mu_hat)^T S^-1 (mu - mu_hat) <= kappa^2 },  S = Sigma / T,
whose closed form gives a second-order cone program:
    max_w  mu_hat^T w - kappa * || S^{1/2} w ||_2 - gamma * w^T Sigma w
    s.t.   sum(w) = 1, w >= 0

The script estimates parameters from 60 months of simulated returns, solves
both problems with cvxpy, and compares them on the true distribution and on
200 repeated samples.
"""
import numpy as np
import cvxpy as cp

rng = np.random.default_rng(0)
N, T, GAMMA = 10, 60, 3.0
mu_true = rng.uniform(0.004, 0.012, N)                       # monthly expected returns
A = rng.normal(size=(N, N)) * 0.02
Sigma_true = A @ A.T + np.diag(rng.uniform(0.001, 0.003, N))

def solve(mu_hat, Sigma_hat, kappa):
    w = cp.Variable(N)
    S_half = np.linalg.cholesky(Sigma_hat / T)
    obj = mu_hat @ w - kappa * cp.norm(S_half.T @ w, 2) - GAMMA * cp.quad_form(w, cp.psd_wrap(Sigma_hat))
    cp.Problem(cp.Maximize(obj), [cp.sum(w) == 1, w >= 0]).solve()
    return w.value

def true_stats(w):
    m, v = mu_true @ w, w @ Sigma_true @ w
    return m, np.sqrt(v), m - GAMMA * v                       # mean, vol, true objective

oracle = solve(mu_true, Sigma_true, 0.0)
results = {"mean-variance": [], "robust (kappa=1)": [], "robust (kappa=2)": []}
weights = {k: [] for k in results}
for _ in range(200):
    R = rng.multivariate_normal(mu_true, Sigma_true, T)
    mu_hat, Sigma_hat = R.mean(0), np.cov(R.T)
    for name, kappa in [("mean-variance", 0.0), ("robust (kappa=1)", 1.0), ("robust (kappa=2)", 2.0)]:
        w = solve(mu_hat, Sigma_hat, kappa)
        results[name].append(true_stats(w)); weights[name].append(w)

print(f"oracle (true parameters): objective {true_stats(oracle)[2]*100:.3f}%")
print(f"\n{'strategy':18s} {'true obj %':>10s} {'obj std':>8s} {'max weight':>10s} {'instability':>11s}")
for name in results:
    r = np.array(results[name]); W = np.array(weights[name])
    instability = np.abs(np.diff(W, axis=0)).sum(1).mean()   # weight change between resamples
    print(f"{name:18s} {r[:, 2].mean()*100:10.3f} {r[:, 2].std()*100:8.3f} "
          f"{W.max(1).mean():10.2f} {instability:11.2f}")
