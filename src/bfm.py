"""Bayesian factorization machines (MCMC, package ``myfm``) with implicit feedback.

Every rating (u, i) is described by the features of two relation blocks
(Rendle 2013; Rendle, Zhang & Koren 2019):
  * user block: one-hot of u, plus with "iu" the normalized set N(u) of items rated
    by u (|N(u)|^-1/2 each), i.e. Bayesian SVD++;
  * item block: one-hot of i, plus with "ii" the normalized set of users who rated i
    ("flipped" implicit feedback), plus with "side" the genre / decade features F.
Optional extra blocks (off by default): "iur" / "iir" same sets valued by the centered
rating (r - global mean) |N|^-1/2, "us" / "ip" user / item log rating count, all computed on ``train`` only.
The rating-valued sets contain the target of each training row (mild for "iur", severe
for "iir" on rarely rated items): they overfit, see experiments/run_fm_variants.py.
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
    F = item_features("data/namesngenre2.npy")
    predict = fit_bfm(train, shape, F, features=("iu", "ii", "side"), rank=10, n_iter=1000)
    predict = fit_bfm_chains(train, shape, F, rank=10, n_iter=2000, seeds=(0, 1, 2))  # averaged chains
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


def implicit_rated(rows, cols, r, shape):
    """CSR (shape) with (r - mean r) / sqrt(row count) at every observed (row, col) position."""
    n = np.bincount(rows, minlength=shape[0])
    vals = (r - r.mean()) / np.sqrt(n[rows])
    return sps.csr_matrix((vals, (rows, cols)), shape=shape)


def standardize(x):
    """Zero-mean, unit-variance columns."""
    return (x - x.mean(0)) / x.std(0)


def log_count(rows, n_rows):
    """(n_rows, 1) CSR of the standardized log(1 + count) of every row id (count-only: a mean
    rating feature would contain each training row's own target and overfit)."""
    return sps.csr_matrix(standardize(np.log1p(np.bincount(rows, minlength=n_rows)))[:, None])


def user_blocks(train, shape, features):
    """List of user-side feature matrices: one-hot, plus implicit items with "iu"
    (optional: "iur" rating-valued implicit items, "us" user log count)."""
    blocks = [one_hot_rows(shape[0])]
    if "iu" in features:
        blocks.append(implicit(train[0], train[1], shape))
    if "iur" in features:
        blocks.append(implicit_rated(train[0], train[1], train[2], shape))
    if "us" in features:
        blocks.append(log_count(train[0], shape[0]))
    return blocks


def item_blocks(train, shape, F, features):
    """List of item-side feature matrices: one-hot, plus implicit users ("ii"), plus F ("side")
    (optional: "iir" rating-valued implicit users, "ip" item log count)."""
    blocks = [one_hot_rows(shape[1])]
    if "ii" in features:
        blocks.append(implicit(train[1], train[0], shape[::-1]))
    if "side" in features:
        blocks.append(sps.csr_matrix(F))
    if "iir" in features:
        blocks.append(implicit_rated(train[1], train[0], train[2], shape[::-1]))
    if "ip" in features:
        blocks.append(log_count(train[1], shape[1]))
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


def fit_model(train, shape, F, features, rank, n_iter, seed=SEED, burn_in=BURN_IN):
    """Run the Gibbs sampler; return (myfm model keeping n_iter - burn_in samples, feature blocks)."""
    u, i, r = train
    blocks = (user_blocks(train, shape, features), item_blocks(train, shape, F, features))
    model = myfm.MyFMRegressor(rank=rank, random_seed=seed)
    model.fit(None, r, X_rel=relations(u, i, blocks), n_iter=n_iter, n_kept_samples=n_iter - burn_in,
              group_shapes=[b.shape[1] for b in blocks[0] + blocks[1]])
    return model, blocks


def fit_bfm(train, shape, F, features=("iu", "ii", "side"), rank=10, n_iter=1000, burn_in=BURN_IN):
    """Fit a Bayesian FM by Gibbs sampling and return predict(u, i) = posterior predictive mean
    (average of the n_iter - burn_in samples drawn after the first burn_in sweeps)."""
    M = posterior_mean(*fit_model(train, shape, F, features, rank, n_iter, burn_in=burn_in))
    return lambda u, i: M[u, i]


def fit_bfm_chains(train, shape, F, features=("iu", "ii", "side"), rank=10, n_iter=2000, seeds=(0, 1, 2),
                   burn_in=BURN_IN):
    """Average the posterior means of independent Gibbs chains (one per seed); return predict(u, i)."""
    M = np.mean([posterior_mean(*fit_model(train, shape, F, features, rank, n_iter, s, burn_in))
                 for s in seeds], 0)
    return lambda u, i: M[u, i]
