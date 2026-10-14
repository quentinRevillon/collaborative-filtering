"""GLocal-K: global and local kernels for matrix completion (Han et al., CIKM 2021).

Item-based autoencoder: every item row r_i (its ratings by the n users, 0 if missing)
goes through 2 hidden layers of 500 sigmoid units and a linear output layer of n units.
Every weight matrix is multiplied entrywise by a local kernel K_jk = max(0, 1 - |a_j - z_k|^2)
of learned 5-d embeddings (Sparse FC). Loss: sum over observed entries of the squared error
+ lam2/2 |W|^2 + lam_s/2 |K|^2 per layer.
  1. Pre-training: the autoencoder is fitted on R by L-BFGS (`iter_per_epoch` iterations
     per epoch, `epochs_p` epochs).
  2. Fine-tuning: a global kernel GK = mean_rows(R') @ C (3 x 3, C learned), built from the
     item means of the current reconstruction R', convolves R: R_hat = relu(conv2d(R, GK)),
     and the (shared) autoencoder is trained on input R_hat for `epochs_f` epochs.
Defaults are the paper's ML-100K settings (lam2 = 20, lam_s = 0.006, 5 L-BFGS iterations
per epoch). Items without train ratings fall back to the biases mu + b_u + b_i.

Usage:
    from src.glocalk import fit_glocalk, fit_glocalk_early
    predict = fit_glocalk(train, shape, epochs_p=30, epochs_f=20)
    predict, curves = fit_glocalk_early(train, shape, valid)   # curves: per-epoch valid RMSE
    r_hat = predict(u, i)
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as Fn

from src.autorec import baseline_matrix
from src.data import clip, rmse

SEED = 0
HIDDEN, KDIM, GK_SIZE = 500, 5, 3


class KernelLayer(nn.Module):
    """Dense layer whose weights are sparsified by a learned finite-support RBF kernel."""

    def __init__(self, n_in, n_out):
        """Xavier weights, zero bias, kernel embeddings ~ N(0, 1e-3)."""
        super().__init__()
        self.W = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_in, n_out)))
        self.b = nn.Parameter(torch.zeros(n_out))
        self.a = nn.Parameter(1e-3 * torch.randn(n_in, KDIM))
        self.z = nn.Parameter(1e-3 * torch.randn(n_out, KDIM))

    def kernel(self):
        """Local kernel max(0, 1 - |a_j - z_k|^2)."""
        return torch.relu(1 - torch.cdist(self.a, self.z) ** 2)

    def forward(self, x):
        """Affine map with the kernelized weights W * K."""
        return x @ (self.W * self.kernel()) + self.b

    def penalty(self, lam2, lam_s):
        """lam2/2 |W|^2 + lam_s/2 |K|^2."""
        return lam2 / 2 * self.W.pow(2).sum() + lam_s / 2 * self.kernel().pow(2).sum()


class GLocalK(nn.Module):
    """Kernelized autoencoder (2 sigmoid hidden layers) plus the global convolution kernel."""

    def __init__(self, n_items, n_users):
        """Layers n_users -> 500 -> 500 -> n_users and the (n_items, 3*3) global kernel map C."""
        super().__init__()
        self.layers = nn.ModuleList([KernelLayer(n_users, HIDDEN), KernelLayer(HIDDEN, HIDDEN),
                                     KernelLayer(HIDDEN, n_users)])
        self.C = nn.Parameter(0.1 * torch.randn(n_items, GK_SIZE ** 2))

    def autoencode(self, X):
        """Reconstruction of the item rows X."""
        h = torch.sigmoid(self.layers[1](torch.sigmoid(self.layers[0](X))))
        return self.layers[2](h)

    def global_input(self, R):
        """relu(conv2d(R, GK)) with GK = item means of the reconstruction of R times C."""
        gk = (self.autoencode(R).mean(1) @ self.C).reshape(1, 1, GK_SIZE, GK_SIZE)
        return torch.relu(Fn.conv2d(R[None, None], gk, padding=GK_SIZE // 2))[0, 0]

    def penalty(self, lam2, lam_s):
        """Sum of the layer penalties."""
        return sum(layer.penalty(lam2, lam_s) for layer in self.layers)


def item_matrix(train, shape):
    """(R, mask) items x users float tensors of the train ratings."""
    R, M = torch.zeros(shape[1], shape[0]), torch.zeros(shape[1], shape[0])
    R[train[1], train[0]] = torch.as_tensor(train[2], dtype=torch.float32)
    M[train[1], train[0]] = 1.0
    return R, M


def lbfgs_epoch(model, params, inputs, R, M, lam2, lam_s, n_iter):
    """Run n_iter L-BFGS iterations on params; `inputs()` builds the autoencoder input."""
    opt = torch.optim.LBFGS(params, max_iter=n_iter, history_size=10, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (M * (R - model.autoencode(inputs())) ** 2).sum() + model.penalty(lam2, lam_s)
        loss.backward()
        return loss

    opt.step(closure)


@torch.no_grad()
def dense_prediction(model, inputs, B, seen):
    """users x items predictions; items without train ratings get the biases B."""
    P = model.autoencode(inputs()).T.numpy().astype(np.float64)
    return np.where(seen[None, :], P, B)


def setup(train, shape):
    """Model, tensors, biases and seen-item mask for one fit."""
    torch.manual_seed(SEED)
    R, M = item_matrix(train, shape)
    seen = np.bincount(train[1], minlength=shape[1]) > 0
    return GLocalK(shape[1], shape[0]), R, M, baseline_matrix(train, shape), seen


def stages(model, R):
    """(pre-training input fn, its params), (fine-tuning input fn, its params)."""
    ae = [p for layer in model.layers for p in layer.parameters()]
    return (lambda: R, ae), (lambda: model.global_input(R), ae + [model.C])


def fit_glocalk(train, shape, epochs_p=30, epochs_f=20, lam2=20.0, lam_s=0.006, iter_per_epoch=5):
    """Pre-train then fine-tune GLocal-K for fixed epoch counts; return predict(u, i)."""
    model, R, M, B, seen = setup(train, shape)
    (pre, p_params), (fine, f_params) = stages(model, R)
    for _ in range(epochs_p):
        lbfgs_epoch(model, p_params, pre, R, M, lam2, lam_s, iter_per_epoch)
    for _ in range(epochs_f):
        lbfgs_epoch(model, f_params, fine, R, M, lam2, lam_s, iter_per_epoch)
    P = dense_prediction(model, fine if epochs_f else pre, B, seen)
    return lambda u, i: P[u, i]


def fit_glocalk_early(train, shape, valid, epochs_p=60, epochs_f=60, lam2=20.0, lam_s=0.006, iter_per_epoch=5):
    """Run both stages, recording the clipped valid RMSE after every epoch.

    Returns (predict of the last epoch, curves) where curves is a list of
    (stage, epoch, valid_rmse); used to choose epochs_p / epochs_f on a validation split.
    """
    model, R, M, B, seen = setup(train, shape)
    curves = []
    for stage, (inputs, params), n in zip(("pre", "fine"), stages(model, R), (epochs_p, epochs_f)):
        for e in range(n):
            lbfgs_epoch(model, params, inputs, R, M, lam2, lam_s, iter_per_epoch)
            P = dense_prediction(model, inputs, B, seen)
            curves.append((stage, e + 1, rmse(clip(P[valid[0], valid[1]]), valid[2])))
            print(curves[-1], flush=True)
    return (lambda u, i: P[u, i]), curves
