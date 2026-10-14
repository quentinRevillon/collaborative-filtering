"""Two-stage models: a base predictor plus a kNN correction of its residuals.

The residuals e_ui of the train ratings are smoothed by a neighbourhood model
(Koren's KNNBaseline with any base instead of biases):
  * axis="item": similarity between items (shrunk Pearson-baseline of the residual
    columns, src/knn.pearson_baseline), correction from the k items rated by u most
    similar to i;
  * axis="user": same on the transposed matrix (similar users who rated i).
    r_hat(u, i) = base(u, i) + sum_j s_ij e_uj / (beta + sum_j s_ij)   (s_ij > 0, top k)

In-sample residuals of a flexible base (the Bayesian FM fits its train ratings to
RMSE ~0.44 vs ~0.87 out of sample) carry almost no signal, so ``fit_oof_knn`` uses
out-of-fold residuals: the base is refitted on n_folds - 1 folds and the residuals
are taken on the held-out fold. The final base is fitted on the whole ``train``.

Usage:
    from src.residual import fit_knn_residual, fit_oof_knn
    predict = fit_knn_residual(train, shape, base, k=40, shrink=100.0, beta=1.0, axis="item")
    predict = fit_oof_knn(train, shape, fit_base, n_folds=5, k=40, shrink=100.0, beta=1.0)
    r_hat = predict(u, i)
where base is a predict(u, i) closure fitted on `train` and fit_base(train) -> predict
fits one (e.g. lambda d: fit_bfm(d, shape, F, n_iter=500)).
"""
import numpy as np

from src.data import to_dense
from src.knn import knn_correction, pearson_baseline


def group_by(keys, n):
    """Indices of `keys` grouped by value 0..n-1 (list of index arrays)."""
    order = np.argsort(keys, kind="stable")
    return np.split(order, np.cumsum(np.bincount(keys, minlength=n))[:-1])


def knn_predict_rows(E, M, S, rows, cols, k, beta):
    """kNN residual correction at (rows, cols): neighbours are columns, similarity S."""
    out = np.zeros(len(rows))
    rated = [np.flatnonzero(m) for m in M]
    for row, idx in enumerate(group_by(rows, E.shape[0])):
        if len(idx) and len(rated[row]):
            sub = S[np.ix_(cols[idx], rated[row])]
            out[idx] = knn_correction(sub, E[row, rated[row]], k, beta)
    return out


def fit_knn_correction(train, shape, e, k=40, shrink=100.0, beta=1.0, axis="item"):
    """Return correct(u, i): kNN smoothing of the residuals e given at the train positions."""
    u, i = train[0], train[1]
    E, M = to_dense(u, i, e, shape), to_dense(u, i, 1.0, shape)
    if axis == "user":
        E, M = E.T, M.T
    S = pearson_baseline(E, M, shrink)

    def correct(pu, pi):
        rows, cols = (pu, pi) if axis == "item" else (pi, pu)
        return knn_predict_rows(E, M, S, rows, cols, k, beta)

    return correct


def fit_knn_residual(train, shape, base, k=40, shrink=100.0, beta=1.0, axis="item"):
    """Return predict(u, i) = base(u, i) + kNN correction of the in-sample train residuals."""
    e = train[2] - base(train[0], train[1])
    correct = fit_knn_correction(train, shape, e, k, shrink, beta, axis)
    return lambda u, i: base(u, i) + correct(u, i)


def fold_ids(n, n_folds=5, seed=0):
    """Random fold index in 0..n_folds-1 of each of n ratings."""
    return np.random.default_rng(seed).integers(0, n_folds, n)


def oof_predictions(train, fit_base, fold):
    """Out-of-fold predictions: each rating predicted by fit_base refitted without its fold."""
    u, i, _ = train
    p = np.zeros(len(u))
    for f in np.unique(fold):
        out = fold == f
        p[out] = fit_base(tuple(a[~out] for a in train))(u[out], i[out])
    return p


def fit_oof_knn(train, shape, fit_base, n_folds=5, k=40, shrink=100.0, beta=1.0, axis="item"):
    """Return predict(u, i) = fit_base(train)(u, i) + kNN correction of out-of-fold residuals."""
    e = train[2] - oof_predictions(train, fit_base, fold_ids(len(train[2]), n_folds))
    correct = fit_knn_correction(train, shape, e, k, shrink, beta, axis)
    base = fit_base(train)
    return lambda u, i: base(u, i) + correct(u, i)
