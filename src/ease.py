"""EASE (Steck 2019): closed-form item-item (or user-user) linear model on residuals.

The dense residual matrix E (users x items, e_ui = r_ui - base(u, i) on observed
entries, 0 elsewhere) is regressed on itself with a zero-diagonal weight matrix:
    min_B |E - E B|^2 + lam |B|^2   s.t. diag(B) = 0
whose solution is B = I - P / diag(P), P = (E^T E + lam I)^-1 (columns divided).
The prediction is r_hat(u, i) = base(u, i) + (E B)[u, i]. With axis="user" the
same model is fitted on E^T (user-user weights, 610 x 610).

Usage:
    from src.ease import fit_ease
    predict = fit_ease(train, shape, base, lam=500.0, axis="item")
    r_hat = predict(u, i)
where base is a predict(u, i) closure fitted on the same `train` (e.g. biases or FM).
"""
import numpy as np

from src.data import to_dense


def residual_matrix(train, shape, base):
    """Dense users x items residuals r - base(u, i) (0 where missing)."""
    u, i, r = train
    return to_dense(u, i, r - base(u, i), shape)


def ease_weights(X, lam):
    """Zero-diagonal EASE weights B (n_cols x n_cols) of the columns of X."""
    P = np.linalg.inv(X.T @ X + lam * np.eye(X.shape[1]))
    B = -P / np.diag(P)
    np.fill_diagonal(B, 0.0)
    return B


def ease_correction(E, lam, axis="item"):
    """Dense users x items EASE reconstruction of the residuals E."""
    if axis == "item":
        return E @ ease_weights(E, lam)
    return (E.T @ ease_weights(E.T, lam)).T


def fit_ease(train, shape, base, lam=500.0, axis="item"):
    """Return predict(u, i) = base(u, i) + EASE correction of the train residuals."""
    C = ease_correction(residual_matrix(train, shape, base), lam, axis)
    return lambda u, i: base(u, i) + C[u, i]
