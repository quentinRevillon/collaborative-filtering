"""Ensemble (stacking) of the Bayesian FM with the other rating predictors.

Members are fitted on 90 % of the ratings, a ridge regression of the held-out 10 %
ratings on the members' clipped predictions learns the blend, then every member is
refitted on all the ratings and combined with those weights (same protocol as
src/blend.py). With ``fwls=True`` the blend is a feature-weighted linear stacking
(Sill et al. 2009): the inputs also contain every prediction multiplied by the
meta-features log(1 + n_u) and log(1 + n_i) (user / item rating counts in the
member's training set), so the weight of each member can depend on the support.

Usage:
    from src.ensemble import fit_ensemble, BEST
    from src.side_info import item_features
    F = item_features("data/namesngenre.npy")
    predict = fit_ensemble(train, shape, F, **BEST)   # train: COO (u, i, r)
    r_hat = predict(u, i)                              # works on millions of pairs (chunked)

Lower-level steps (used by experiments/run_ensemble.py on cached predictions):
    fits = candidate_fits(F)                           # {name: fit(train, shape)}
    X = features(P, meta(counts(train, shape), u, i), fwls)
"""
from functools import partial

import numpy as np
from sklearn.linear_model import Ridge

from src.autorec import fit_autorec
from src.baselines import fit_biases
from src.bfm import fit_bfm
from src.blend import fit_all, stack_predictions
from src.bpmf import fit_bpmf
from src.data import split
from src.knn import fit_item_knn
from src.mf import fit_als
from src.side_info import fit_side_biases
from src.soft_impute import fit_soft_impute

FRAC, SEED = 0.1, 0
CHUNK = 500_000                     # pairs per predict call when blending a full matrix
BEST = dict(names=("fm", "knn", "fm_iu_ii"), alpha=10.0, fwls=False)  # selected on valid CV, results/ensemble_results.csv


def fit_user_knn(train, shape, **hp):
    """User-based KNNBaseline: item-kNN on the transposed ratings."""
    predict = fit_item_knn((train[1], train[0], train[2]), shape[::-1], **hp)
    return lambda u, i: predict(i, u)


def candidate_fits(F):
    """{name: fit(train, shape) -> predict} of every candidate ensemble member."""
    return {
        "fm": partial(fit_bfm, F=F, features=("iu", "ii", "side"), rank=10, n_iter=2000),
        "fm_ii": partial(fit_bfm, F=F, features=("ii",), rank=10, n_iter=1000),
        "fm_iu_ii": partial(fit_bfm, F=F, features=("iu", "ii"), rank=10, n_iter=1000),
        "fm_mf": partial(fit_bfm, F=F, features=(), rank=10, n_iter=1000),
        "fm_r5": partial(fit_bfm, F=F, features=("iu", "ii", "side"), rank=5, n_iter=1000),
        "fm_r20": partial(fit_bfm, F=F, features=("iu", "ii", "side"), rank=20, n_iter=1000),
        "bpmf": partial(fit_bpmf, rank=20, alpha=2.0, n_samples=300, burn_in=100),
        "knn": partial(fit_item_knn, k=40, shrink=100.0, beta=0.2, lam=2.0),
        "uknn": partial(fit_user_knn, k=40, shrink=100.0, beta=0.2, lam=2.0),
        "side": partial(fit_side_biases, F=F, lam=2.0, lam_w=10.0, pop=True),
        "si": partial(fit_soft_impute, tau=10.0, lam=2.0),
        "als": partial(fit_als, rank=10, lam_f=12.0, lam_b=2.0),
        "biases": partial(fit_biases, lam=2.0),
        "autorec": partial(fit_autorec, hidden=500, lam=30.0, item=True, residual=True, epochs=181),
    }


def counts(train, shape):
    """(log(1 + n_u) per user, log(1 + n_i) per item) of the ratings `train`."""
    return (np.log1p(np.bincount(train[0], minlength=shape[0])),
            np.log1p(np.bincount(train[1], minlength=shape[1])))


def meta(cnt, u, i):
    """(n, 2) meta-features [log(1 + n_u), log(1 + n_i)] of the pairs (u, i)."""
    return np.column_stack([cnt[0][u], cnt[1][i]])


def features(P, M, fwls):
    """Blend inputs: predictions P (n, k), plus P * each meta column and M itself if `fwls`."""
    if not fwls:
        return P
    return np.hstack([P] + [P * M[:, [m]] for m in range(M.shape[1])] + [M])


def blend_inputs(predicts, cnt, u, i, fwls):
    """Blend inputs of fitted members {name: predict} at the pairs (u, i)."""
    return features(stack_predictions(predicts, u, i), meta(cnt, u, i), fwls)


def chunked(predict):
    """predict(u, i) evaluated by chunks of CHUNK pairs (bounded memory)."""
    return lambda u, i: np.concatenate([predict(u[k:k + CHUNK], i[k:k + CHUNK]) for k in range(0, len(u), CHUNK)])


def fit_ensemble(train, shape, F, names, alpha=1.0, fwls=False):
    """Learn the ridge blend of members `names` on a 10 % split, refit them on all of `train`."""
    fits = {n: candidate_fits(F)[n] for n in names}
    part, valid = split(train, FRAC, SEED)
    X = blend_inputs(fit_all(fits, part, shape), counts(part, shape), valid[0], valid[1], fwls)
    ridge = Ridge(alpha=alpha).fit(X, valid[2])
    predicts, cnt = fit_all(fits, train, shape), counts(train, shape)
    return chunked(lambda u, i: ridge.predict(blend_inputs(predicts, cnt, u, i, fwls)))
