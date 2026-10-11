"""Side information (genres, release decade) as a prior on the item biases.

Model: r_ui ~ mu + b_u + b_i with a hierarchical prior b_i ~ N(f_i . w, 1/lam):
the item bias is shrunk towards a genre/decade-based value f_i . w instead of
0, which mostly helps movies with few (or no) training ratings. With
pop=True the prior also uses log(1 + n_i), the item's train popularity
(rarely rated movies are rated lower on average). Fitted by
alternating closed-form updates of b_i, w (ridge, penalty lam_w) and b_u.

Usage:
    from src.side_info import item_features, fit_side_biases
    F = item_features("data/namesngenre.npy")          # (n_items, n_features)
    predict = fit_side_biases(train, shape, F, lam=2.0, lam_w=1.0, pop=True)
    r_hat = predict(u, i)
"""
import re

import numpy as np


def parse_year(title):
    """Release year at the end of 'Title (Year)', or -1 if absent."""
    m = re.search(r"\((\d{4})\)\s*$", title)
    return int(m.group(1)) if m else -1


def one_hot(labels):
    """One-hot encode a list of hashable labels (columns in sorted order)."""
    cats = sorted(set(labels))
    idx = np.array([cats.index(x) for x in labels])
    return np.eye(len(cats))[idx]


def genre_matrix(genres):
    """Multi-hot matrix of the '|'-separated genre strings."""
    split = [g.split("|") for g in genres]
    cats = sorted({x for gs in split for x in gs})
    return np.array([[c in gs for c in cats] for gs in split], dtype=float)


def item_features(path):
    """Genre multi-hot + decade one-hot features of every movie."""
    meta = np.load(path, allow_pickle=True)
    decades = [parse_year(t) // 10 * 10 if parse_year(t) > 0 else -1 for t in meta[:, 0]]
    return np.hstack([genre_matrix(meta[:, 1]), one_hot(decades)])


def side_biases(train, shape, F, lam, lam_w, n_iter=20):
    """Fit mu, b_u, b_i with b_i shrunk towards F w; returns (mu, bu, bi, w)."""
    u, i, r = train
    mu = r.mean()
    nu = np.bincount(u, minlength=shape[0])
    ni = np.bincount(i, minlength=shape[1])
    bu, w = np.zeros(shape[0]), np.zeros(F.shape[1])
    A = F.T @ F + lam_w / lam * np.eye(F.shape[1])
    for _ in range(n_iter):
        bi = (np.bincount(i, r - mu - bu[u], minlength=shape[1]) + lam * (F @ w)) / (lam + ni)
        w = np.linalg.solve(A, F.T @ bi)
        bu = np.bincount(u, r - mu - bi[i], minlength=shape[0]) / (lam + nu)
    return mu, bu, bi, w


def popularity_features(train, shape):
    """log(1 + n_i) and its square, n_i = number of train ratings of item i."""
    n = np.log1p(np.bincount(train[1], minlength=shape[1]))
    return np.column_stack([n, n ** 2])


def fit_side_biases(train, shape, F, lam=2.0, lam_w=1.0, pop=False):
    """Predict mu + b_u + b_i with genre/decade-informed item biases.

    If `pop`, the prior also uses the item popularity in `train` (log count).
    """
    if pop:
        F = np.hstack([F, popularity_features(train, shape)])
    mu, bu, bi, _ = side_biases(train, shape, F, lam, lam_w)
    return lambda u, i: mu + bu[u] + bi[i]
