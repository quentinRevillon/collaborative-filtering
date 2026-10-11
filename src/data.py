"""Data utilities shared by all models and experiments.

Ratings matrices are users x items numpy arrays with NaN for missing entries.
Models work on COO triplets ``(u, i, r)``.

Usage:
    from src.data import load_coo, split, evaluate
    train, shape = load_coo("data/ratings_train.npy")
    fit, valid = split(train, frac=0.1, seed=0)
    rmse, acc = evaluate(predict, valid)   # predict(u, i) -> ratings
"""
import numpy as np

LO, HI = 0.5, 5.0


def to_coo(R):
    """Convert a NaN-masked dense matrix into COO triplets (u, i, r)."""
    u, i = np.nonzero(np.isfinite(R))
    return u, i, R[u, i]


def load_coo(path):
    """Load a NaN-masked .npy ratings matrix as COO triplets and its shape."""
    R = np.load(path)
    return to_coo(R), R.shape


def split(data, frac, seed=0):
    """Randomly split COO ratings into (fit, valid), `frac` of them in valid."""
    perm = np.random.default_rng(seed).permutation(len(data[2]))
    cut = int(len(perm) * frac)
    return tuple(a[perm[cut:]] for a in data), tuple(a[perm[:cut]] for a in data)


def to_dense(u, i, v, shape):
    """Scatter values into a dense zero-filled matrix."""
    M = np.zeros(shape)
    M[u, i] = v
    return M


def clip(pred):
    """Clip predictions to the valid rating range [0.5, 5]."""
    return np.clip(pred, LO, HI)


def rmse(pred, r):
    """Root mean squared error."""
    return float(np.sqrt(np.mean((pred - r) ** 2)))


def accuracy(pred, r):
    """Fraction of predictions equal to the target after rounding to 0.5."""
    return float(np.mean(np.round(pred * 2) / 2 == r))


def evaluate(predict, data):
    """Return (rmse, accuracy) of the clipped predictions of `predict` on `data`."""
    p = clip(predict(data[0], data[1]))
    return rmse(p, data[2]), accuracy(p, data[2])


def complete(predict, shape):
    """Fill a dense users x items matrix with the clipped predictions of `predict`."""
    u, i = np.indices(shape).reshape(2, -1)
    return clip(predict(u, i)).reshape(shape)
