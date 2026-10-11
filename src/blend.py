"""Linear blending (ridge regression) of several rating predictors.

Protocol: every model is fitted on 90 % of the train ratings, the blend
weights are learned by ridge regression of the held-out 10 % ratings on the
models' (clipped) predictions, then every model is refitted on the full train
set and combined with those weights.

Usage:
    from functools import partial
    from src.blend import fit_blend
    fits = {"biases": partial(fit_biases, lam=2.0), "si": partial(fit_soft_impute, tau=10.0)}
    predict = fit_blend(train, shape, fits, alpha=1.0)
    r_hat = predict(u, i)

Lower-level steps (reuse already fitted predictors):
    ridge = learn_weights(predicts_90, valid, alpha)   # predicts: {name: predict}
    predict = combine(predicts_full, ridge)
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict

from src.data import clip, rmse, split


def stack_predictions(predicts, u, i):
    """(n, n_models) matrix of clipped predictions, columns in dict order."""
    return np.column_stack([clip(p(u, i)) for p in predicts.values()])


def learn_weights(predicts, valid, alpha=1.0):
    """Ridge regression of the valid ratings on the models' predictions."""
    return Ridge(alpha=alpha).fit(stack_predictions(predicts, valid[0], valid[1]), valid[2])


def cv_rmse(predicts, valid, alpha=1.0, folds=5):
    """Honest RMSE of the blend estimated by K-fold CV on the valid ratings."""
    X = stack_predictions(predicts, valid[0], valid[1])
    pred = cross_val_predict(Ridge(alpha=alpha), X, valid[2], cv=folds)
    return rmse(clip(pred), valid[2])


def combine(predicts, ridge):
    """Blended predictor ridge(stack of predictions)."""
    return lambda u, i: ridge.predict(stack_predictions(predicts, u, i))


def fit_all(fits, train, shape):
    """Fit every model of {name: fit(train, shape)} and return {name: predict}."""
    return {name: fit(train, shape) for name, fit in fits.items()}


def fit_blend(train, shape, fits, alpha=1.0, frac=0.1, seed=0):
    """Learn blend weights on a `frac` split, refit models on all of `train`."""
    fit_part, valid = split(train, frac, seed)
    ridge = learn_weights(fit_all(fits, fit_part, shape), valid, alpha)
    return combine(fit_all(fits, train, shape), ridge)
