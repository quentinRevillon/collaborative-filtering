"""Bayesian probabilistic matrix factorization (BPMF) with biases, by Gibbs sampling.

Model (Salakhutdinov & Mnih, ICML 2008), extended with user/item biases:
    r_ui ~ N(mu + b_u + b_i + p_u . q_i, 1 / alpha)
Each user row w_u = [p_u, b_u] (and item row w_i = [q_i, b_i]) is a (rank+1)-vector
with a Gaussian prior N(m_U, Lam_U^-1) whose mean and precision get a Normal-Wishart
hyperprior, so factors and biases are regularized jointly and the regularization is
learnt from the data. Given the items, every user row is a Bayesian linear regression
on the design [q_i, 1] with target r_ui - mu - b_i, sampled exactly; all rows are
sampled at once with batched Cholesky factorizations (torch). The noise precision
alpha is either fixed or sampled (Gamma prior) when ``alpha=None``.

Predictions are the average of the dense prediction matrices of the samples drawn
after ``burn_in`` sweeps (Monte-Carlo estimate of the posterior predictive mean).

Usage:
    from src.bpmf import fit_bpmf, gibbs_samples, running_means
    predict = fit_bpmf(train, shape, rank=10, n_samples=200, burn_in=50, alpha=None)
    r_hat = predict(u, i)
    for M in gibbs_samples(train, shape, rank=10, alpha=2.0): ...   # one dense matrix per sweep
    for M in running_means(gibbs_samples(...), burn_in=50): ...     # posterior mean estimates
"""
from itertools import islice

import numpy as np
import torch
from scipy.stats import wishart

SEED = 0
INIT_STD = 0.1
BETA0 = 2.0          # Normal-Wishart: strength of the prior on the row mean (prior mean 0)
A0, B0 = 1.0, 1.0    # Gamma(A0, rate B0) prior on the noise precision alpha
CHUNK = 8192         # observations per chunk when accumulating per-row Gram matrices


def gram(X, rows, n):
    """Per-row Gram matrices sum_{obs of row} x x^T, shape (n, d, d), accumulated by chunks."""
    A = torch.zeros(n, X.shape[1], X.shape[1], dtype=X.dtype)
    for s in range(0, len(rows), CHUNK):
        x = X[s:s + CHUNK]
        A.index_add_(0, rows[s:s + CHUNK], x[:, :, None] * x[:, None, :])
    return A


def sample_hyper(W, rng):
    """Draw (mean, precision) of the row prior from its Normal-Wishart posterior given rows W."""
    n, d = W.shape
    xbar = W.mean(0)
    S = (W - xbar).T @ (W - xbar)
    scale_inv = np.eye(d) + S + BETA0 * n / (BETA0 + n) * np.outer(xbar, xbar)
    Lam = wishart(df=d + n, scale=np.linalg.inv(scale_inv)).rvs(random_state=rng)
    mean = rng.multivariate_normal(n * xbar / (BETA0 + n), np.linalg.inv((BETA0 + n) * Lam))
    return torch.from_numpy(mean), torch.from_numpy(Lam)


def sample_rows(obs, prior, alpha, gen):
    """Sample every row w ~ N(A^-1 b, A^-1), A = Lam + alpha sum x x^T, b = Lam m + alpha sum x t.

    obs = (X, t, rows, n): (n_obs, d) design, (n_obs,) targets, (n_obs,) row index of each
    observation and number of rows; prior = (m, Lam) from sample_hyper. Rows without
    observations are drawn from the prior.
    """
    X, t, rows, n = obs
    m, Lam = prior
    A = Lam + alpha * gram(X, rows, n)
    b = (Lam @ m).expand(n, -1).clone().index_add_(0, rows, alpha * X * t[:, None])
    L = torch.linalg.cholesky(A)
    mean = torch.cholesky_solve(b[:, :, None], L)
    z = torch.randn(mean.shape, dtype=X.dtype, generator=gen)
    return (mean + torch.linalg.solve_triangular(L.mT, z, upper=True))[:, :, 0]


def design(W):
    """Split rows [w, b] of one side into the other side's design [w, 1] and offset b."""
    return torch.cat([W[:, :-1], torch.ones(len(W), 1, dtype=W.dtype)], 1), W[:, -1]


def half_step(W, other, obs, alpha, rng, gen):
    """Resample all rows W given the other side's rows; obs = (rows, cols, centered ratings)."""
    rows, cols, r = obs
    X, off = design(other)
    prior = sample_hyper(W.numpy(), rng)
    return sample_rows((X[cols], r - off[cols], rows, len(W)), prior, alpha, gen)


def dense(mu, U, V):
    """Dense prediction matrix mu + b_u + b_i + p_u . q_i of one sample (numpy)."""
    return (mu + U[:, :-1] @ V[:, :-1].T + U[:, -1:] + V[:, -1][None]).numpy()


def sample_alpha(M, train, rng):
    """Draw the noise precision from its Gamma posterior given the sample's predictions M."""
    u, i, r = train
    sse = float(np.sum((r - M[u, i]) ** 2))
    return rng.gamma(A0 + len(r) / 2, 1.0 / (B0 + sse / 2))


def gibbs_samples(train, shape, rank, alpha=None):
    """Infinite generator of the dense prediction matrix after each Gibbs sweep.

    alpha=None samples the noise precision (starting from 1), a float keeps it fixed.
    """
    u, i, r = train
    rng, gen = np.random.default_rng(SEED), torch.Generator().manual_seed(SEED)
    mu = r.mean()
    tu, ti, tr = torch.from_numpy(u), torch.from_numpy(i), torch.from_numpy(r - mu)
    U = torch.from_numpy(rng.normal(0.0, INIT_STD, (shape[0], rank + 1)))
    V = torch.from_numpy(rng.normal(0.0, INIT_STD, (shape[1], rank + 1)))
    a = 1.0 if alpha is None else alpha
    while True:
        U = half_step(U, V, (tu, ti, tr), a, rng, gen)
        V = half_step(V, U, (ti, tu, tr), a, rng, gen)
        M = dense(mu, U, V)
        if alpha is None:
            a = sample_alpha(M, train, rng)
        yield M


def running_means(samples, burn_in):
    """Yield the running average of the matrices of `samples` after skipping `burn_in` of them."""
    total = 0.0
    for k, M in enumerate(islice(samples, burn_in, None), start=1):
        total = total + M
        yield total / k


def from_dense(M):
    """predict(u, i) reading a dense prediction matrix."""
    return lambda u, i: M[u, i]


def fit_bpmf(train, shape, rank=10, n_samples=200, burn_in=50, alpha=None):
    """Fit BPMF with biases and return predict(u, i) = posterior mean over n_samples draws."""
    means = running_means(gibbs_samples(train, shape, rank, alpha), burn_in)
    return from_dense(next(islice(means, n_samples - 1, None)))
