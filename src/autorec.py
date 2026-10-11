"""AutoRec (Sedhain et al., WWW 2015) and its kernelized variant (Sparse FC, Muller et al. 2018).

A one-hidden-layer autoencoder reconstructs, for every item (I-AutoRec) or
every user (U-AutoRec), its vector of observed ratings:
    h = sigmoid(W^T x + b),   x_hat = V^T h + c
The loss is the squared error on observed entries only plus an L2 penalty:
    1/2 sum_obs (x - x_hat)^2 + lam/2 (|W|^2 + |V|^2)
With ``residual=True`` the autoencoder works on bias residuals
r - (mu + b_u + b_i) and the biases are added back at prediction time.
With ``kernel_dim > 0`` every weight matrix is multiplied entrywise by a learned
finite-support kernel K_jk = max(0, 1 - |a_j - z_k|^2), a_j, z_k in R^kernel_dim
(Sparse FC / local kernel of GLocal-K), whose squared norm is penalized by LAM_S.

Training is full-batch Adam (one step = one epoch, sparse matmul encoder, dense
decoder gathered on the observed entries). Users/items without train ratings
fall back to the biases mu + b_u + b_i.

Usage:
    from src.autorec import fit_autorec, fit_autorec_early
    predict = fit_autorec(train, shape, hidden=500, lam=10.0, item=True, residual=True, epochs=200)
    predict, curves = fit_autorec_early(train, shape, valid, hidden=500, lam=10.0)
    n_epochs = 1 + int(np.argmin(curves[:, 1]))   # curves: (epochs, 2) train / valid RMSE
    r_hat = predict(u, i)
"""
import numpy as np
import torch
from torch import nn

from src.baselines import biases
from src.data import clip, rmse

torch.set_num_threads(4)
LR = 3e-3
BIAS_LAM = 2.0
LAM_S = 0.006
SEED = 0


class AutoRec(nn.Module):
    """One-hidden-layer autoencoder, optionally with kernelized (Sparse FC) weights."""

    def __init__(self, n_in, hidden, kernel_dim, mean):
        """Xavier-initialized weights; output bias at the mean target; kernel embeddings near 0."""
        super().__init__()
        self.W = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_in, hidden)))
        self.V = nn.Parameter(nn.init.xavier_uniform_(torch.empty(hidden, n_in)))
        self.b = nn.Parameter(torch.zeros(hidden))
        self.c = nn.Parameter(torch.full((n_in,), float(mean)))
        dims = (n_in, hidden, hidden, n_in) if kernel_dim else ()
        self.emb = nn.ParameterList([nn.Parameter(1e-3 * torch.randn(n, kernel_dim)) for n in dims])

    def kernels(self):
        """Encoder and decoder kernels max(0, 1 - |a_j - z_k|^2) (empty list if not kernelized)."""
        e = self.emb
        return [torch.relu(1 - torch.cdist(e[k], e[k + 1]) ** 2) for k in range(0, len(e), 2)]

    def weights(self):
        """Effective (W, V, kernels): weights multiplied by their kernel when kernelized."""
        K = self.kernels()
        return (self.W * K[0], self.V * K[1], K) if K else (self.W, self.V, K)

    def forward(self, X, W, V):
        """Dense reconstruction sigmoid(X W + b) V + c of the sparse input rows X."""
        return torch.sigmoid(torch.sparse.mm(X, W) + self.b) @ V + self.c


def baseline_matrix(train, shape):
    """Dense users x items matrix of the bias predictions mu + b_u + b_i."""
    mu, bu, bi = biases(train, shape, BIAS_LAM)
    return mu + bu[:, None] + bi[None, :]


def setup(train, shape, hidden, item, residual, kernel_dim):
    """Build the model, the training tensors and the prediction context."""
    torch.manual_seed(SEED)
    B = baseline_matrix(train, shape)
    u, i, r = train
    y = r - B[u, i] if residual else r
    rows, cols = (i, u) if item else (u, i)
    n_rows, n_in = shape[::-1] if item else shape
    X = torch.sparse_coo_tensor(np.vstack([rows, cols]), y, (n_rows, n_in), dtype=torch.float32).coalesce()
    batch = dict(X=X, flat=torch.as_tensor(rows * n_in + cols), y=torch.as_tensor(y, dtype=torch.float32))
    seen = np.bincount(rows, minlength=n_rows) > 0
    seen = np.broadcast_to(seen[None, :] if item else seen[:, None], shape)
    ctx = dict(item=item, residual=residual, B=B, seen=seen)
    return AutoRec(n_in, hidden, kernel_dim, y.mean()), batch, ctx


def loss_step(model, opt, batch, lam):
    """One full-batch Adam step on the masked loss; returns the loss value."""
    W, V, K = model.weights()
    p = model(batch["X"], W, V).flatten()[batch["flat"]]
    reg = lam / 2 * (W.pow(2).sum() + V.pow(2).sum()) + LAM_S / 2 * sum(k.pow(2).sum() for k in K)
    loss = 0.5 * (p - batch["y"]).pow(2).sum() + reg
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss)


@torch.no_grad()
def reconstruct(model, batch, ctx):
    """Autoencoder output as a users x items float32 array (without biases or fallback)."""
    out = model(batch["X"], *model.weights()[:2]).numpy()
    return out.T if ctx["item"] else out


def entries(out, ctx, data):
    """Predictions at the (u, i) entries of `data`; rows without train ratings get the biases."""
    u, i = data[0], data[1]
    b = ctx["B"][u, i]
    p = out[u, i] + b if ctx["residual"] else out[u, i]
    return np.where(ctx["seen"][u, i], p, b)


def dense(out, ctx):
    """Dense users x items predictions; rows without train ratings get the biases."""
    P = out + ctx["B"] if ctx["residual"] else out.astype(np.float64)
    return np.where(ctx["seen"], P, ctx["B"])


def fit_autorec(train, shape, hidden=500, lam=10.0, item=True, residual=True, kernel_dim=0, epochs=200):
    """Train AutoRec for a fixed number of full-batch epochs and return predict(u, i)."""
    model, batch, ctx = setup(train, shape, hidden, item, residual, kernel_dim)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(epochs):
        loss_step(model, opt, batch, lam)
    P = dense(reconstruct(model, batch, ctx), ctx)
    return lambda u, i: P[u, i]


def fit_autorec_early(train, shape, valid, max_epochs=600, patience=30, **hp):
    """Train with early stopping on `valid` (u, i, r); return (predict at best epoch, curves).

    `curves` is an (epochs_run, 2) array of clipped train / valid RMSE, row e
    being measured after e + 1 updates; `hp` are the keyword arguments of fit_autorec
    (except epochs). The best number of epochs is 1 + argmin(curves[:, 1]).
    """
    hp = dict(dict(hidden=500, lam=10.0, item=True, residual=True, kernel_dim=0), **hp)
    model, batch, ctx = setup(train, shape, hp["hidden"], hp["item"], hp["residual"], hp["kernel_dim"])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    curves, best = [], None
    for epoch in range(max_epochs):
        loss_step(model, opt, batch, hp["lam"])
        out = reconstruct(model, batch, ctx)
        curves.append([rmse(clip(entries(out, ctx, d)), d[2]) for d in (train, valid)])
        best = out if curves[-1][1] <= min(c[1] for c in curves) else best
        if epoch - int(np.argmin([c[1] for c in curves])) >= patience:
            break
    P = dense(best, ctx)
    return (lambda u, i: P[u, i]), np.array(curves)
