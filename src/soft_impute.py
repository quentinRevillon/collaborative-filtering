"""Soft-Impute (Mazumder, Hastie & Tibshirani, 2010) on bias residuals.

The residuals r - (mu + b_u + b_i) are completed by iterating
    Z <- SVT_tau( P_Omega(X) + P_Omega^perp(Z) )
where SVT_tau soft-thresholds the singular values at level tau. Since the
matrix has few rows (610 users), the SVD is obtained from the eigen-
decomposition of the small Gram matrix X X^T (610 x 610), which is ~10x faster
than a full SVD of the 610 x 4980 matrix. Iterations stop when the relative
change of Z falls below `tol`; a tau path reuses the previous solution as warm
start.

Usage:
    from src.soft_impute import fit_soft_impute, fit_soft_impute_path
    predict = fit_soft_impute(train, shape, tau=10.0, lam=5.0)
    r_hat = predict(u, i)
    predicts = fit_soft_impute_path(train, shape, taus=[20, 14, 10], lam=5.0)
"""
import numpy as np

from src.baselines import fit_biases
from src.data import to_dense


def svt(X, tau, max_rank=None):
    """Singular value soft-thresholding of a wide matrix X at level tau.

    Uses X = U S V^T  =>  SVT(X) = U diag((s - tau)_+ / s) U^T X, with (U, s^2)
    the eigenpairs of X X^T. Optionally keeps at most `max_rank` components.
    """
    ev, U = np.linalg.eigh(X @ X.T)
    s = np.sqrt(np.maximum(ev[::-1], 0.0))
    U = U[:, ::-1]
    keep = s > tau
    if max_rank is not None:
        keep[max_rank:] = False
    Uk = U[:, keep]
    return (Uk * ((s[keep] - tau) / s[keep])) @ (Uk.T @ X)


def soft_impute(X, M, tau, Z, tol=1e-4, max_iter=100, max_rank=None):
    """Run Soft-Impute iterations from the warm start Z until convergence.

    X: observed residuals (0 where missing), M: boolean observation mask.
    """
    for _ in range(max_iter):
        Z_new = svt(np.where(M, X, Z), tau, max_rank)
        delta = np.sum((Z_new - Z) ** 2) / max(np.sum(Z ** 2), 1e-12)
        Z = Z_new
        if delta < tol:
            break
    return Z


def residual_matrix(train, shape, lam):
    """Return the bias predictor, dense residual matrix and observation mask."""
    base = fit_biases(train, shape, lam)
    X = to_dense(train[0], train[1], train[2] - base(train[0], train[1]), shape)
    M = to_dense(train[0], train[1], 1.0, shape) > 0
    return base, X, M


def make_predict(base, Z):
    """Predictor mu + b_u + b_i + Z[u, i]."""
    return lambda u, i: base(u, i) + Z[u, i]


def fit_soft_impute(train, shape, tau=10.0, lam=5.0, max_rank=None):
    """Soft-Impute on bias residuals with threshold `tau` and bias ridge `lam`."""
    base, X, M = residual_matrix(train, shape, lam)
    Z = soft_impute(X, M, tau, np.zeros(shape), max_rank=max_rank)
    return make_predict(base, Z)


def fit_soft_impute_path(train, shape, taus, lam=5.0):
    """Fit Soft-Impute for each tau (sorted decreasingly) with warm starts.

    Returns a dict {tau: predict}.
    """
    base, X, M = residual_matrix(train, shape, lam)
    Z, out = np.zeros(shape), {}
    for tau in sorted(taus, reverse=True):
        Z = soft_impute(X, M, tau, Z)
        out[tau] = make_predict(base, Z)
    return out
