"""Submitted model: Bayesian FMs on user-normalized ratings, item kNN, gradient-boosting correction.

1. User normalization: for every user, mean m_u and standard deviation s_u of his ratings, shrunk
   towards the global values with SHRINK pseudo-ratings; the Bayesian FM is fitted on
   z_ui = (r_ui - m_u) / s_u and its predictions are mapped back with m_u + s_u * z. The FM has a
   single noise level for everybody: on z, users whose ratings are very spread out weigh less.
2. Base models: posterior means of N_CHAINS chains of that FM (and, if USE_RAW, of the FM on raw
   ratings), item kNN on bias residuals (src/knn.py, fit_item_knn_damped).
3. Linear stack of the base models + gradient boosting (scikit-learn) on its residual, with
   features of the rating (base predictions, disagreement between chains, user / movie
   statistics, genres, decade).
   Stack and boosting need out-of-fold predictions of the base models: they were computed once by
   5-fold cross-validation on ratings_train + ratings_test (the platform input) and are versioned
   in data/stack_train.npz; the stack and the boosting are refitted on them here (a few seconds).
   If the ratings given to fit() are not exactly those of data/stack_train.npz, the boosting is
   skipped and the linear stack uses the FALLBACK weights learnt in cross-validation.

Usage:
    from src.final_stack import fit_final_stack
    predict = fit_final_stack(train, shape)      # train: COO (u, i, r)
    r_hat = predict(u, i)
"""
import re
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from src.bfm import fit_model, posterior_mean
from src.knn import fit_item_knn_damped
from src.side_info import item_features

ROOT = Path(__file__).resolve().parent.parent
GENRES = ROOT / "data" / "namesngenre2.npy"
STACK_TRAIN = ROOT / "data" / "stack_train.npz"

FM = dict(features=("iu", "ii", "side"), rank=20, n_iter=238, burn_in=20)
N_CHAINS = 5
SHRINK = 20.0
USE_RAW = True
KNN = dict(k=20, shrink=100.0, damp=0.5, lam_u=5.0, lam_i=3.0)
GBDT = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.0,
            early_stopping=False, random_state=0)
# Linear stack (intercept, FM normalized, FM raw, kNN), least squares on the out-of-fold predictions:
# used only if data/stack_train.npz does not match the input ratings
FALLBACK = (0.0948, 0.8136, 0.161, 0.0005)
BATCH = 50        # users per batch when predicting the full matrix


def user_scale(train, shape, shrink=SHRINK):
    """Shrunk per-user mean and standard deviation of the ratings."""
    u, _, r = train
    mu, sigma = r.mean(), r.std()
    n = np.bincount(u, minlength=shape[0])
    m = (np.bincount(u, r, minlength=shape[0]) + shrink * mu) / (n + shrink)
    dev = np.bincount(u, (r - m[u]) ** 2, minlength=shape[0])
    s = np.sqrt((dev + shrink * sigma ** 2) / (n + shrink))
    return m, s


def fm_chains(train, shape, F, normalized):
    """Dense posterior means of N_CHAINS Bayesian FM chains (seeds 0..N_CHAINS-1)."""
    u, i, r = train
    if normalized:
        m, s = user_scale(train, shape)
        data = (u, i, (r - m[u]) / s[u])
    else:
        data = train
    chains = np.array([posterior_mean(*fit_model(data, shape, F, FM["features"], FM["rank"], FM["n_iter"],
                                                 seed, FM["burn_in"])) for seed in range(N_CHAINS)])
    if normalized:
        chains = m[None, :, None] + s[None, :, None] * chains
    return chains


def rating_stats(train, shape):
    """User / movie statistics of the ratings (same definitions as in the cross-validation)."""
    u, i, r = train
    n_u = np.bincount(u, minlength=shape[0]).astype(float)
    n_i = np.bincount(i, minlength=shape[1]).astype(float)
    mean_u = np.bincount(u, r, minlength=shape[0]) / np.maximum(n_u, 1)
    std_u = np.sqrt(np.bincount(u, (r - mean_u[u]) ** 2, minlength=shape[0]) / np.maximum(n_u, 1))
    half_u = np.bincount(u, (r * 2) % 2 == 1, minlength=shape[0]) / np.maximum(n_u, 1)
    mean_i = np.where(n_i > 0, np.bincount(i, r, minlength=shape[1]) / np.maximum(n_i, 1), r.mean())
    std_i = np.sqrt(np.bincount(i, (r - mean_i[i]) ** 2, minlength=shape[1]) / np.maximum(n_i, 1))
    return dict(n_u=n_u, n_i=n_i, mean_u=mean_u, std_u=std_u, half_u=half_u, mean_i=mean_i, std_i=std_i)


def movie_side():
    """Release decade (numeric) and genre multi-hot of every movie, from namesngenre2.npy."""
    meta = np.load(GENRES, allow_pickle=False)
    years = [re.search(r"\((\d{4})\)\s*$", t) for t in meta[:, 0]]
    decades = np.array([int(y.group(1)) // 10 * 10 if y else -1 for y in years], dtype=float)
    genres = [g.split("|") for g in meta[:, 1]]
    names = sorted({g for gs in genres for g in gs})
    return decades, np.array([[g in gs for g in names] for gs in genres], dtype=float)


def features(base, chain_std, stats, u, i, decades, G):
    """Boosting features of the ratings (u, i); base: columns of base-model predictions."""
    return np.column_stack([base, chain_std, np.log1p(stats["n_i"][i]), np.log1p(stats["n_u"][u]),
                            stats["std_u"][u], stats["mean_u"][u], stats["half_u"][u], stats["mean_i"][i],
                            stats["std_i"][i], decades[i], G[i]])


def matches_stack_train(train, shape, saved):
    """True if `train` holds exactly the ratings of data/stack_train.npz."""
    u, i, r = train
    if len(r) != len(saved["truth"]):
        return False
    a, b = np.zeros(shape), np.zeros(shape)
    a[u, i], b[saved["user"], saved["item"]] = r, saved["truth"]
    return bool(np.array_equal(a, b))


def fit_final_stack(train, shape):
    """Fit the submitted model and return predict(u, i)."""
    F = item_features(GENRES)
    norm = fm_chains(train, shape, F, normalized=True)
    dense = [norm.mean(0)]
    if USE_RAW:
        dense.append(fm_chains(train, shape, F, normalized=False).mean(0))
    knn = fit_item_knn_damped(train, shape, **KNN)
    chain_std = norm.std(0)
    decades, G = movie_side()
    stats = rating_stats(train, shape)

    saved = np.load(STACK_TRAIN) if STACK_TRAIN.exists() else None
    if saved is not None and matches_stack_train(train, shape, saved):
        X = np.column_stack([saved["fm_norm"]] + ([saved["fm_raw"]] if USE_RAW else []) + [saved["knn"]])
        y = saved["truth"]
        w = np.linalg.lstsq(np.column_stack([np.ones(len(y)), X]), y, rcond=None)[0]
        oof_stats = {k: saved[k] for k in ("n_u", "n_i", "mean_u", "std_u", "half_u", "mean_i", "std_i")}
        gb = HistGradientBoostingRegressor(**GBDT).fit(
            features(X, saved["chain_std"], oof_stats, saved["user"], saved["item"], decades, G),
            y - (w[0] + X @ w[1:]))
    else:
        print("data/stack_train.npz does not match the input ratings: linear stack only")
        w, gb = np.asarray(FALLBACK), None

    def predict(u, i):
        out = np.empty(len(u))
        for start in range(0, len(u), BATCH * shape[1]):
            sl = slice(start, start + BATCH * shape[1])
            uu, ii = u[sl], i[sl]
            base = np.column_stack([d[uu, ii] for d in dense] + [knn(uu, ii)])
            pred = w[0] + base @ w[1:]
            if gb is not None:
                pred = pred + gb.predict(features(base, chain_std[uu, ii], stats, uu, ii, decades, G))
            out[sl] = pred
        return out

    return predict
