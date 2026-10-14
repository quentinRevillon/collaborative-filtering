"""Gradient-boosted correction of a base model from its out-of-fold residuals.

A small gradient-boosted tree model g learns the residual of the base model from
meta-features of the rating (feature-weighted stacking, Sill et al. 2009):
    r_hat(u, i) = p_ui + g(p_ui, log(1+n_u), m_u, s_u, log(1+n_i), m_i, s_i)
with p_ui the base prediction, n the rating counts, m the means and s the standard
deviations of the user's / item's ratings, shrunk towards the global moments with
`prior` pseudo-ratings. g is trained on out-of-fold residuals: the base is refitted
on n_folds - 1 folds, and the support statistics of a rating are computed on the same
folds, so the training features carry exactly the information available at
prediction time (in-sample residuals of the Bayesian FM are ~0.44 RMSE vs ~0.87
out of sample and would be useless). At prediction time the base is fitted on the
whole train set and the statistics use every train rating.

Usage:
    from src.residual_gbdt import fit_residual_gbdt
    fit_base = lambda d: fit_bfm(d, shape, F, n_iter=2000)    # final base model
    fit_fold = lambda d: fit_bfm(d, shape, F, n_iter=500)     # cheaper out-of-fold base
    predict = fit_residual_gbdt(train, shape, fit_base, fit_fold, n_folds=5)
    r_hat = predict(u, i)
Lower level (to share base fits): fit_gbdt_correction(train, shape, p_oof, fold) -> correct(p, u, i).
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from src.residual import fold_ids, oof_predictions

GBDT = dict(learning_rate=0.02, max_iter=300, min_samples_leaf=100, max_depth=3,
            l2_regularization=1.0, random_state=0)


def shrunk_moments(keys, n, r, w, prior, mu, var):
    """Per-key (log(1 + count), shrunk mean, shrunk std) of the ratings r weighted by w."""
    c = np.bincount(keys, w, minlength=n)
    m = (np.bincount(keys, w * r, minlength=n) + prior * mu) / (c + prior)
    q = (np.bincount(keys, w * r ** 2, minlength=n) + prior * (var + mu ** 2)) / (c + prior)
    return np.column_stack([np.log1p(c), m, np.sqrt(np.maximum(q - m ** 2, 0.0))])


def support_features(train, shape, w, u, i, prior=10.0):
    """(n, 6) user and item statistics at (u, i) computed from the train ratings weighted by w."""
    tu, ti, r = train
    mu, var = r.mean(), r.var()
    return np.hstack([shrunk_moments(tu, shape[0], r, w, prior, mu, var)[u],
                      shrunk_moments(ti, shape[1], r, w, prior, mu, var)[i]])


def oof_features(train, shape, p_oof, fold, prior=10.0):
    """Meta-features of the train ratings, statistics computed without each rating's fold."""
    u, i, _ = train
    X = np.zeros((len(u), 6))
    for f in np.unique(fold):
        out = fold == f
        X[out] = support_features(train, shape, (~out).astype(float), u[out], i[out], prior)
    return np.column_stack([p_oof, X])


def fit_gbdt_correction(train, shape, p_oof, fold, prior=10.0):
    """Return correct(p, u, i): GBDT residual estimate for base predictions p at (u, i)."""
    g = HistGradientBoostingRegressor(**GBDT).fit(oof_features(train, shape, p_oof, fold, prior), train[2] - p_oof)
    w = np.ones(len(train[2]))
    return lambda p, u, i: g.predict(np.column_stack([p, support_features(train, shape, w, u, i, prior)]))


def fit_residual_gbdt(train, shape, fit_base, fit_fold, n_folds=5, prior=10.0):
    """Return predict(u, i) = base(u, i) + GBDT correction learned on out-of-fold residuals."""
    fold = fold_ids(len(train[2]), n_folds)
    correct = fit_gbdt_correction(train, shape, oof_predictions(train, fit_fold, fold), fold, prior)
    base = fit_base(train)

    def predict(u, i):
        p = base(u, i)
        return p + correct(p, u, i)

    return predict
