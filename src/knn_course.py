"""Neighborhood-based CF exactly as in the course: weighted average of raw neighbor ratings.

    r_hat(u, i) = sum_{j in N_k(i; u)} s_ij r_uj / sum_j |s_ij|     (item-based)

s_ij is the Pearson correlation between items i and j over their common users (ratings
centered by each item's mean), optionally shrunk by n_ij / (n_ij + shrink). Only positive
similarities are kept as neighbors. When no neighbor exists, the course says to "use a
baseline": we use the global mean. User-based = the same on the transposed ratings.

Usage:
    from src.knn_course import fit_course_knn
    predict = fit_course_knn(train, shape, k=40, shrink=0.0, axis="item")
    r_hat = predict(u, i)
"""
import numpy as np

from src.data import to_dense
from src.knn import knn_correction, pearson_baseline


def column_centered(train, shape):
    """Dense ratings minus each column's (item's) mean, 0 where missing."""
    u, i, r = train
    means = np.bincount(i, r, minlength=shape[1]) / np.maximum(np.bincount(i, minlength=shape[1]), 1)
    return to_dense(u, i, r - means[i], shape)


def fit_item_course_knn(train, shape, k, shrink):
    """Item-based course formula; returns predict(u, i)."""
    R = to_dense(train[0], train[1], train[2], shape)
    M = to_dense(train[0], train[1], 1.0, shape)
    S = np.nan_to_num(pearson_baseline(column_centered(train, shape), M, shrink))  # 0/0 when shrink = 0 and no co-rating
    rated = [np.flatnonzero(row) for row in M]
    mu = train[2].mean()

    def predict(u, i):
        out = np.full(len(u), mu)
        order = np.argsort(u, kind="stable")
        for user, idx in enumerate(np.split(order, np.cumsum(np.bincount(u, minlength=shape[0]))[:-1])):
            if len(idx) and len(rated[user]):
                avg = knn_correction(S[np.ix_(i[idx], rated[user])], R[user, rated[user]], k)
                out[idx] = np.where(avg > 0, avg, mu)
        return out

    return predict


def fit_course_knn(train, shape, k=40, shrink=0.0, axis="item"):
    """Course kNN formula, item-based or user-based (transposed); returns predict(u, i)."""
    if axis == "item":
        return fit_item_course_knn(train, shape, k, shrink)
    predict = fit_item_course_knn((train[1], train[0], train[2]), shape[::-1], k, shrink)
    return lambda u, i: predict(i, u)
