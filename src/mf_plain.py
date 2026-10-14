"""Plain matrix factorization of the course, R ~ I U^T, applied after removing a baseline.

    r_ui ~ base(u, i) + <p_u, q_i>,   min sum_obs (r_ui - base - p_u.q_i)^2 + lam (|P|^2 + |Q|^2)

base is fixed before the factorization: "none" (raw ratings, exactly the course),
"mean" (global mean removed) or "biases" (mean + regularized user and movie biases
removed, src/baselines.py). Two solvers: alternating least squares (exact ridge solution
for each user, then each movie) and stochastic gradient descent (mini-batches).
src/mf.py instead learns the biases jointly with the factors.

Usage:
    from src.mf_plain import fit_plain_als, fit_plain_sgd
    predict = fit_plain_als(train, shape, base="biases", rank=10, lam=10.0)
    predict = fit_plain_sgd(train, shape, base="none", rank=10, lam=0.05, lr=0.01, n_epochs=30)
"""
import numpy as np

from src.baselines import fit_biases
from src.mf import BATCH, SEED, group, init_factors, ridge_rows

BIAS_LAM = 2.0


def fit_base(train, shape, base):
    """predict(u, i) of the fixed baseline "none", "mean" or "biases"."""
    if base == "none":
        return lambda u, i: np.zeros(len(u))
    if base == "mean":
        mu = train[2].mean()
        return lambda u, i: np.full(len(u), mu)
    return fit_biases(train, shape, BIAS_LAM)


def with_factors(b, P, Q):
    """predict(u, i) = b(u, i) + <p_u, q_i>."""
    return lambda u, i: b(u, i) + np.einsum("nk,nk->n", P[u], Q[i])


def fit_plain_als(train, shape, base, rank, lam, n_iter=15):
    """Alternating least squares on r - base; returns predict(u, i)."""
    u, i, r = train
    b = fit_base(train, shape, base)
    t = r - b(u, i)
    rng = np.random.default_rng(SEED)
    P, Q = init_factors(shape[0], rank, rng), init_factors(shape[1], rank, rng)
    gu, gi = group(u, shape[0]), group(i, shape[1])
    reg_u, reg_i = np.full((shape[0], rank), lam), np.full((shape[1], rank), lam)
    for _ in range(n_iter):
        P = ridge_rows(gu, Q[i], t, reg_u)
        Q = ridge_rows(gi, P[u], t, reg_i)
    return with_factors(b, P, Q)


def fit_plain_sgd(train, shape, base, rank, lam, lr=0.01, n_epochs=30):
    """Mini-batch stochastic gradient descent on r - base; returns predict(u, i)."""
    u, i, r = train
    b = fit_base(train, shape, base)
    t = r - b(u, i)
    rng = np.random.default_rng(SEED)
    P, Q = init_factors(shape[0], rank, rng), init_factors(shape[1], rank, rng)
    for _ in range(n_epochs):
        for idx in np.array_split(rng.permutation(len(t)), max(len(t) // BATCH, 1)):
            pu, qi = P[u[idx]], Q[i[idx]]
            e = t[idx] - np.einsum("nk,nk->n", pu, qi)
            np.add.at(P, u[idx], lr * (e[:, None] * qi - lam * pu))
            np.add.at(Q, i[idx], lr * (e[:, None] * pu - lam * qi))
    return with_factors(b, P, Q)
