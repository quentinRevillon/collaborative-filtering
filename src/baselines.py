"""Baseline predictors: global mean and regularized biases mu + b_u + b_i.

Every ``fit_*`` function takes COO triplets ``train = (u, i, r)`` and the matrix
shape, and returns a ``predict(u, i)`` closure.

Usage:
    from src.baselines import fit_biases
    predict = fit_biases(train, shape, lam=2.0)
    r_hat = predict(u, i)
"""
import numpy as np


def fit_global_mean(train, shape):
    """Predict the global mean rating everywhere."""
    mu = train[2].mean()
    return lambda u, i: np.full(len(u), mu)


def biases(train, shape, lam, n_iter=20):
    """Fit mu, b_u, b_i by alternating closed-form ridge updates.

    Minimizes sum (r - mu - b_u - b_i)^2 + lam * (|b_u|^2 + |b_i|^2).
    """
    u, i, r = train
    mu = r.mean()
    nu = np.bincount(u, minlength=shape[0])
    ni = np.bincount(i, minlength=shape[1])
    bu = np.zeros(shape[0])
    for _ in range(n_iter):
        bi = np.bincount(i, r - mu - bu[u], minlength=shape[1]) / np.maximum(lam + ni, 1e-12)  # 0 if unrated
        bu = np.bincount(u, r - mu - bi[i], minlength=shape[0]) / np.maximum(lam + nu, 1e-12)
    return mu, bu, bi


def fit_biases(train, shape, lam=2.0):
    """Predict mu + b_u + b_i with ridge-regularized biases."""
    mu, bu, bi = biases(train, shape, lam)
    return lambda u, i: mu + bu[u] + bi[i]
