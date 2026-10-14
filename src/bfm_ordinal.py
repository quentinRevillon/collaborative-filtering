"""Bayesian factorization machine with an ordered probit likelihood (myfm.MyFMOrderedProbit).

The 10 rating levels 0.5, 1, ..., 5 are classes k = 0..9. Given the FM score f (same
features as src/bfm.py) and the sampled cutpoints c_0 < ... < c_8,
    P(y <= k) = Phi(c_k - f),   so   E[r] = 0.5 + 0.5 * sum_k Phi(f - c_k).
The prediction is the posterior mean of E[r] (average over Gibbs samples). As in
src/bfm.py the score of every (user, item) pair is computed densely sample by sample;
on the dense matrix E[r] is a smooth increasing function of f, evaluated by linear
interpolation on a fine grid (error < 1e-5), and only every `thin`-th kept sample is used.

Usage:
    from src.bfm_ordinal import fit_ordinal_bfm
    from src.side_info import item_features
    F = item_features("data/namesngenre.npy")
    predict = fit_ordinal_bfm(train, shape, F, features=("iu", "ii", "side"), rank=10, n_iter=2000)
    r_hat = predict(u, i)
    model, blocks = fit_ordinal_model(train, shape, F, ("iu", "ii", "side"), rank=10, n_iter=500)
    E = expected_at(model, blocks, u, i)          # posterior expected rating at positions (u, i)
"""
import myfm
import numpy as np
from scipy.special import ndtr

from src.bfm import BURN_IN, SEED, chunk_sum, item_blocks, relations, split_sample, stacked, user_blocks

THIN = 5
GRID = 4001          # interpolation nodes for the dense expected rating


def to_class(r):
    """Ratings 0.5..5 -> ordinal classes 0..9."""
    return np.round(2 * r - 1).astype(np.int64)


def expected_rating(S, cutpoints):
    """Expected rating 0.5 + 0.5 * sum_k Phi(S - c_k) for scores S (any shape)."""
    return 0.5 + 0.5 * sum(ndtr(S - c) for c in cutpoints)


def expected_rating_grid(S, cutpoints):
    """expected_rating(S, cutpoints) by linear interpolation on a GRID-point grid over [min S, max S]."""
    grid = np.linspace(S.min(), S.max(), GRID)
    return np.interp(S, grid, expected_rating(grid, cutpoints))


def fit_ordinal_model(train, shape, F, features, rank, n_iter, seed=SEED):
    """Run the ordered-probit Gibbs sampler; return (myfm model, feature blocks)."""
    u, i, r = train
    blocks = (user_blocks(train, shape, features), item_blocks(train, shape, F, features))
    model = myfm.MyFMOrderedProbit(rank=rank, random_seed=seed)
    model.fit(None, to_class(r), X_rel=relations(u, i, blocks), n_iter=n_iter,
              n_kept_samples=n_iter - BURN_IN, group_shapes=[b.shape[1] for b in blocks[0] + blocks[1]])
    return model, blocks


def kept(model, thin):
    """Every `thin`-th kept posterior sample."""
    return list(model.predictor_.samples)[::thin]


def expected_at(model, blocks, u, i, thin=1):
    """Posterior mean of E[r] at the positions (u, i)."""
    Xu, Xi = stacked(blocks)
    samples = kept(model, thin)
    total = 0.0
    for fm in samples:
        w0, (gu, au), (gi, ai) = split_sample(fm, Xu, Xi)
        total = total + expected_rating(w0 + gu[u] + gi[i] + np.sum(au[u] * ai[i], 1), fm.cutpoints[0])
    return total / len(samples)


def expected_dense(model, blocks, thin=THIN):
    """Posterior mean of E[r] over the whole users x items matrix."""
    Xu, Xi = stacked(blocks)
    samples = kept(model, thin)
    return sum(expected_rating_grid(chunk_sum([split_sample(fm, Xu, Xi)]), fm.cutpoints[0])
               for fm in samples) / len(samples)


def fit_ordinal_bfm(train, shape, F, features=("iu", "ii", "side"), rank=10, n_iter=2000, thin=THIN):
    """Fit an ordered-probit Bayesian FM and return predict(u, i) = posterior expected rating."""
    M = expected_dense(*fit_ordinal_model(train, shape, F, features, rank, n_iter), thin=thin)
    return lambda u, i: M[u, i]
