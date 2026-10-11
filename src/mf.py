"""Biased matrix factorization r_ui ~ mu + b_u + b_i + p_u . q_i.

Two solvers, both with separate ridge penalties on factors (lam_f) and biases (lam_b):
  * ALS: masked alternating least squares. Each half-step solves one ridge
    regression per user (resp. item) on its observed ratings only, the bias being
    learnt jointly through a constant-1 column appended to the fixed factors.
    ``wr=True`` scales the factor penalty by the number of ratings (ALS-WR, Zhou et al. 2008).
  * SGD: Funk-SVD trained by shuffled mini-batch stochastic gradient descent.

``als_steps`` / ``sgd_steps`` are generators yielding a ``predict`` closure after each
iteration / epoch; ``fit_als`` / ``fit_sgd`` return the last one and ``history``
turns a generator into learning curves.

Usage:
    from src.mf import fit_als, fit_sgd, als_steps, history
    predict = fit_als(train, shape, rank=10, lam_f=10.0, lam_b=5.0, n_iter=15)
    predict = fit_sgd(train, shape, rank=10, lam_f=0.05, lam_b=0.01, lr=0.01, n_epochs=40)
    curves = history(als_steps(train, shape, 10, 0.0, 0.0, 15), train, valid)  # [(train, valid) RMSE]
"""
import numpy as np
import torch

from src.data import evaluate

SEED = 0
INIT_STD = 0.1
BATCH = 128


def init_factors(n, rank, rng):
    """Draw an (n, rank) factor matrix from N(0, INIT_STD^2)."""
    return rng.normal(0.0, INIT_STD, (n, rank))


def make_predict(mu, bu, bi, P, Q):
    """Return predict(u, i) = mu + b_u + b_i + <p_u, q_i>."""
    return lambda u, i: mu + bu[u] + bi[i] + np.einsum("nk,nk->n", P[u], Q[i])


def group(rows, n):
    """Split the indices of `rows` into n index arrays, one per row value (possibly empty)."""
    order = np.argsort(rows, kind="stable")
    return np.split(order, np.cumsum(np.bincount(rows, minlength=n))[:-1])


def penalty(groups, rank, lam_f, lam_b, wr):
    """Stack per-row diagonal penalties (n, rank+1): lam_f on factors, lam_b on the bias.

    With `wr`, the factor penalty of each row is multiplied by its number of ratings
    (at least 1); the bias penalty stays lam_b, as in the biases baseline.
    """
    n = np.maximum([len(g) for g in groups], 1) if wr else np.ones(len(groups))
    return np.column_stack([np.outer(n, np.full(rank, lam_f)), np.full(len(groups), lam_b)])


def ridge_rows(groups, X, t, reg):
    """Solve, for every group g, (X_g^T X_g + diag(reg_g)) w_g = X_g^T t_g in one batched call.

    torch's batched solver is used: numpy's is pathologically slow on many tiny systems
    with multithreaded OpenBLAS.

    X: (n_obs, d) fixed design, t: (n_obs,) targets, reg: (n_groups, d). Returns (n_groups, d).
    """
    A = np.stack([X[g].T @ X[g] for g in groups])
    b = np.stack([X[g].T @ t[g] for g in groups])
    A[:, np.arange(X.shape[1]), np.arange(X.shape[1])] += reg
    return torch.linalg.solve(torch.from_numpy(A), torch.from_numpy(b)).numpy()


def als_steps(train, shape, rank, lam_f, lam_b, n_iter, wr=False):
    """Yield predict after each ALS sweep (users then items) of biased masked ALS."""
    u, i, r = train
    rng = np.random.default_rng(SEED)
    mu = r.mean()
    P, Q = init_factors(shape[0], rank, rng), init_factors(shape[1], rank, rng)
    bu, bi = np.zeros(shape[0]), np.zeros(shape[1])
    gu, gi = group(u, shape[0]), group(i, shape[1])
    reg_u, reg_i = penalty(gu, rank, lam_f, lam_b, wr), penalty(gi, rank, lam_f, lam_b, wr)
    ones = np.ones((len(r), 1))
    for _ in range(n_iter):
        W = ridge_rows(gu, np.hstack([Q[i], ones]), r - mu - bi[i], reg_u)
        P, bu = W[:, :rank], W[:, rank]
        W = ridge_rows(gi, np.hstack([P[u], ones]), r - mu - bu[u], reg_i)
        Q, bi = W[:, :rank], W[:, rank]
        yield make_predict(mu, bu, bi, P, Q)


def sgd_epoch(train, params, lam_f, lam_b, lr, rng):
    """Run one shuffled mini-batch SGD epoch in place on params = (mu, bu, bi, P, Q).

    Per-rating gradients of (r - r_hat)^2 / 2 + penalties are summed within a batch
    with np.add.at, so a batch of size 1 is exactly Funk's SGD.
    """
    u, i, r = train
    mu, bu, bi, P, Q = params
    for idx in np.array_split(rng.permutation(len(r)), max(len(r) // BATCH, 1)):
        bu_, bi_, pu, qi = bu[u[idx]], bi[i[idx]], P[u[idx]], Q[i[idx]]
        e = r[idx] - mu - bu_ - bi_ - np.einsum("nk,nk->n", pu, qi)
        np.add.at(bu, u[idx], lr * (e - lam_b * bu_))
        np.add.at(bi, i[idx], lr * (e - lam_b * bi_))
        np.add.at(P, u[idx], lr * (e[:, None] * qi - lam_f * pu))
        np.add.at(Q, i[idx], lr * (e[:, None] * pu - lam_f * qi))


def sgd_steps(train, shape, rank, lam_f, lam_b, lr, n_epochs):
    """Yield predict after each SGD epoch of biased Funk-SVD."""
    rng = np.random.default_rng(SEED)
    params = (train[2].mean(), np.zeros(shape[0]), np.zeros(shape[1]),
              init_factors(shape[0], rank, rng), init_factors(shape[1], rank, rng))
    for _ in range(n_epochs):
        sgd_epoch(train, params, lam_f, lam_b, lr, rng)
        yield make_predict(*params)


def last(steps):
    """Exhaust a generator and return its last item."""
    for item in steps:
        pass
    return item


def fit_als(train, shape, rank, lam_f, lam_b, n_iter=15, wr=False):
    """Fit biased ALS and return predict(u, i)."""
    return last(als_steps(train, shape, rank, lam_f, lam_b, n_iter, wr))


def fit_sgd(train, shape, rank, lam_f, lam_b, lr, n_epochs):
    """Fit biased Funk-SVD by mini-batch SGD and return predict(u, i)."""
    return last(sgd_steps(train, shape, rank, lam_f, lam_b, lr, n_epochs))


def history(steps, train, valid):
    """Return the list of (train RMSE, valid RMSE) after each step of a *_steps generator."""
    return [(evaluate(p, train)[0], evaluate(p, valid)[0]) for p in steps]
