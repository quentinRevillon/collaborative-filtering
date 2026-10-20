"""Item-based kNN on bias residuals (Koren's KNNBaseline).

Similarity between items i and j is the Pearson-baseline correlation of the
residuals e_uj = r_uj - (mu + b_u + b_j) over their common users U_ij,
shrunk by the number of co-ratings:
    s_ij = n_ij / (n_ij + shrink) * sum_U e_ui e_uj / sqrt(sum_U e_ui^2 sum_U e_uj^2)
The prediction for (u, i) uses the k items rated by u most similar to i,
keeping positive similarities only:
    r_hat = b_ui + sum_j s_ij e_uj / (beta + sum_j s_ij)
and falls back to the baseline b_ui when sum_j s_ij is zero. beta = 0 is the
course/Surprise formula; beta > 0 shrinks the correction towards the baseline
when the neighbours are few or weakly similar (very useful at ~1 % density).

fit_item_knn_damped is the variant used by the submitted stack (generate.py): it keeps
negative similarities too, with weights s_ij / (damp + sum_j |s_ij|), and a baseline with
separate user / item regularization:
    r_hat = b_ui + sum_j s_ij e_uj / (damp + sum_j |s_ij|)

Usage:
    from src.knn import fit_item_knn, fit_item_knn_damped
    predict = fit_item_knn(train, shape, k=40, shrink=100.0, beta=0.2, lam=2.0)
    predict = fit_item_knn_damped(train, shape, k=20, shrink=100.0, damp=0.5, lam_u=5.0, lam_i=3.0)
    r_hat = predict(u, i)
"""
import numpy as np

from src.baselines import biases, fit_biases
from src.data import to_dense


def pearson_baseline(E, M, shrink):
    """Shrunk Pearson-baseline similarity between the columns of E.

    E: users x items residuals (0 where missing), M: float mask of observations.
    """
    E, M = E.astype(np.float32), M.astype(np.float32)
    num = E.T @ E
    sq = (E ** 2).T @ M  # sq[i, j] = sum over common users of e_ui^2
    n = M.T @ M
    S = num / np.sqrt(np.maximum(sq * sq.T, 1e-12)) * (n / (n + shrink))
    np.fill_diagonal(S, 0.0)
    return S


def knn_correction(S_sub, e_rated, k, beta=0.0):
    """kNN residual correction for a batch of query items of one user.

    S_sub: (n_queries, n_rated) similarities to the items rated by the user,
    e_rated: (n_rated,) residuals of those ratings, beta: denominator
    regularization. Returns (n_queries,).
    """
    if S_sub.shape[1] > k:
        top = np.argpartition(-S_sub, k - 1, axis=1)[:, :k]
        S_sub = np.take_along_axis(S_sub, top, axis=1)
        e = e_rated[top]
    else:
        e = np.broadcast_to(e_rated, S_sub.shape)
    W = np.maximum(S_sub, 0.0)
    den = W.sum(1)
    return np.where(den > 0, (W * e).sum(1) / np.where(den > 0, den + beta, 1.0), 0.0)


def fit_item_knn(train, shape, k=40, shrink=100.0, beta=0.0, lam=2.0):
    """Item-based KNNBaseline with shrunk Pearson-baseline similarity."""
    base = fit_biases(train, shape, lam)
    E = to_dense(train[0], train[1], train[2] - base(train[0], train[1]), shape)
    M = to_dense(train[0], train[1], 1.0, shape)
    S = pearson_baseline(E, M, shrink)
    rated = [np.flatnonzero(row) for row in M]

    def predict(u, i):
        out = base(u, i)
        order = np.argsort(u, kind="stable")
        bounds = np.cumsum(np.bincount(u, minlength=shape[0]))
        for user, idx in enumerate(np.split(order, bounds[:-1])):
            if len(idx) and len(rated[user]):
                sub = S[np.ix_(i[idx], rated[user])]
                out[idx] += knn_correction(sub, E[user, rated[user]], k, beta)
        return out

    return predict


def fit_item_knn_damped(train, shape, k=20, shrink=100.0, damp=0.5, lam_u=5.0, lam_i=3.0):
    """Item kNN on bias residuals, signed similarities, damped denominator (submitted variant).

    The baseline uses 10 alternating updates, as in va/preliminary (src/models/knn.py) where
    the stack weights were learnt: neighbours are picked among many near-tied similarities, so
    even tiny changes of the residuals change some predictions.
    """
    mu, bu, bi = biases(train, shape, lam_u, n_iter=10, lam_i=lam_i)
    base = lambda u, i: mu + bu[u] + bi[i]
    E = to_dense(train[0], train[1], train[2] - base(train[0], train[1]), shape).astype(np.float32)
    M = to_dense(train[0], train[1], 1.0, shape)
    S = pearson_baseline(E, M, shrink)
    rated = [np.flatnonzero(row) for row in M]

    def predict(u, i):
        out = base(u, i)
        order = np.argsort(u, kind="stable")
        bounds = np.cumsum(np.bincount(u, minlength=shape[0]))
        for user, idx in enumerate(np.split(order, bounds[:-1])):
            if len(idx) and len(rated[user]):
                sub = S[np.ix_(i[idx], rated[user])]
                e = E[user, rated[user]]
                if sub.shape[1] > k:
                    top = np.argpartition(-sub, k - 1, axis=1)[:, :k]
                    sub, e = np.take_along_axis(sub, top, axis=1), e[top]
                den = np.abs(sub).sum(1) + damp
                out[idx] += np.where(den > 0, (sub * e).sum(1) / np.where(den > 0, den, 1.0), 0.0)
        return out

    return predict
