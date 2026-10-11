"""Bayesian factorization machines (MCMC, package ``myfm``) with implicit feedback.

Every rating (u, i) is described by the features of two relation blocks
(Rendle 2013; Rendle, Zhang & Koren 2019):
  * user block: one-hot of u, plus with "iu" the normalized set N(u) of items rated
    by u (|N(u)|^-1/2 each), i.e. Bayesian SVD++;
  * item block: one-hot of i, plus with "ii" the normalized set of users who rated i
    ("flipped" implicit feedback), plus with "side" the genre / decade features F.
Without extra features the model is Bayesian MF with biases. Every feature group has its
own Normal-Gamma hyperprior (``group_shapes``). Implicit sets only use the rating positions
of the model input ``train`` (no leakage of the evaluation positions).

The Gaussian (regression) FM is used. Since every feature belongs either to the user or
to the item block, the FM score of one sample splits into
    w0 + g_u + g_i + <a_u, b_i>,   a_u = x_u V, g_u = x_u w + (|x_u V|^2 - x_u^2 V^2) / 2
(same for items), so the posterior mean of the whole users x items matrix is computed
densely, sample by sample, instead of calling myfm's row-wise predictor on 3M rows.

Usage:
    from src.bfm import fit_bfm
    from src.side_info import item_features
    F = item_features("data/namesngenre.npy")
    predict = fit_bfm(train, shape, F, features=("iu", "ii", "side"), rank=10, n_iter=1000)
    r_hat = predict(u, i)
    model, blocks = fit_model(train, shape, F, ("iu",), rank=10, n_iter=500)
    S = sample_scores(model, blocks, u, i)          # per-sample predictions, for MCMC curves
"""
import myfm
import numpy as np
import scipy.sparse as sps

SEED = 0
BURN_IN = 50
CHUNK = 100          # samples per dense matmul in posterior_mean


def one_hot_rows(n):
    """(n, n) sparse identity: one-hot encoding of n ids."""
    return sps.identity(n, format="csr")


def implicit(rows, cols, shape):
    """CSR (shape) with 1/sqrt(row count) at every observed (row, col) position."""
    n = np.bincount(rows, minlength=shape[0])
    vals = 1.0 / np.sqrt(n[rows])
    return sps.csr_matrix((vals, (rows, cols)), shape=shape)


def user_blocks(train, shape, features):
    """List of user-side feature matrices: one-hot, plus implicit items with "iu"."""
    blocks = [one_hot_rows(shape[0])]
    if "iu" in features:
        blocks.append(implicit(train[0], train[1], shape))
    return blocks


def item_blocks(train, shape, F, features):
    """List of item-side feature matrices: one-hot, plus implicit users ("ii"), plus F ("side")."""
    blocks = [one_hot_rows(shape[1])]
    if "ii" in features:
        blocks.append(implicit(train[1], train[0], shape[::-1]))
    if "side" in features:
        blocks.append(sps.csr_matrix(F))
    return blocks


def relations(u, i, blocks):
    """myfm relation blocks for the rating positions (u, i) given (user_blocks, item_blocks)."""
    Xu, Xi = stacked(blocks)
    return [myfm.RelationBlock(u, Xu), myfm.RelationBlock(i, Xi)]


def side_terms(X, w, V):
    """Per-row linear + intra-block pairwise FM terms g and latent sums a of one block X."""
    a = X @ V
    g = X @ w + 0.5 * (np.sum(a ** 2, 1) - X.multiply(X) @ np.sum(V ** 2, 1))
    return g, a


def split_sample(fm, Xu, Xi):
    """Decompose one posterior sample into (w0, (g_u, a_u), (g_i, a_i)) (features: user then item)."""
    n = Xu.shape[1]
    return fm.w0, side_terms(Xu, fm.w[:n], fm.V[:n]), side_terms(Xi, fm.w[n:], fm.V[n:])


def chunk_sum(parts):
    """Sum of the dense users x items score matrices of a few split samples (one matmul)."""
    w0, gu, gi = (sum(x) for x in zip(*[(w0, gu, gi) for w0, (gu, _), (gi, _) in parts]))
    return w0 + gu[:, None] + gi[None] + np.hstack([p[1][1] for p in parts]) @ np.hstack([p[2][1] for p in parts]).T


def stacked(blocks):
    """Concatenate the user and item feature blocks into two CSR matrices."""
    return tuple(sps.hstack(b, format="csr") for b in blocks)


def posterior_mean(model, blocks):
    """Average over the kept samples of the dense users x items score matrices."""
    Xu, Xi = stacked(blocks)
    samples = list(model.predictor_.samples)
    chunks = [samples[k:k + CHUNK] for k in range(0, len(samples), CHUNK)]
    return sum(chunk_sum([split_sample(fm, Xu, Xi) for fm in c]) for c in chunks) / len(samples)


def sample_scores(model, blocks, u, i):
    """(n_samples, n) predictions of every kept sample at the positions (u, i)."""
    Xu, Xi = stacked(blocks)
    parts = (split_sample(fm, Xu, Xi) for fm in model.predictor_.samples)
    return np.array([w0 + gu[u] + gi[i] + np.sum(au[u] * ai[i], 1) for w0, (gu, au), (gi, ai) in parts])


def fit_model(train, shape, F, features, rank, n_iter):
    """Run the Gibbs sampler; return (myfm model keeping n_iter - BURN_IN samples, feature blocks)."""
    u, i, r = train
    blocks = (user_blocks(train, shape, features), item_blocks(train, shape, F, features))
    model = myfm.MyFMRegressor(rank=rank, random_seed=SEED)
    model.fit(None, r, X_rel=relations(u, i, blocks), n_iter=n_iter, n_kept_samples=n_iter - BURN_IN,
              group_shapes=[b.shape[1] for b in blocks[0] + blocks[1]])
    return model, blocks


def fit_bfm(train, shape, F, features=("iu", "ii", "side"), rank=10, n_iter=1000):
    """Fit a Bayesian FM by Gibbs sampling and return predict(u, i) = posterior predictive mean."""
    M = posterior_mean(*fit_model(train, shape, F, features, rank, n_iter))
    return lambda u, i: M[u, i]
