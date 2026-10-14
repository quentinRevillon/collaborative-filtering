"""GC-MC: graph convolutional matrix completion (van den Berg, Kipf & Welling, 2017).

The ratings form a bipartite user-item graph with one edge type per rating level r
(0.5, 1, ..., 5: 10 levels).
  * Encoder (one graph-convolution layer, "stack" accumulation): for every level r the
    hidden state of a user concatenates sum_{j in N_r(u)} E_r[j] / sqrt(|N(u)| |N(j)|) over
    the items j it rated r (same for items with user embeddings), followed by a ReLU and a
    linear dense layer. Ordinal weight sharing: E_r = sum_{s <= r} T_s. Item side features
    (genres, decade) enter the dense layer through a small ReLU layer. Node dropout drops all
    the messages of a node with probability `dropout` during training.
  * Decoder: bilinear softmax over the levels, logit_r(u, v) = h_u^T (sum_b a_rb P_b) h_v with
    2 basis matrices P_b; the prediction is the expected rating sum_r r p(r | u, v).
Trained full batch with Adam (lr 0.01) on the cross-entropy of the observed levels.
The paper's ML-100K sizes (hidden 500 / 75) overfit here (valid RMSE ~0.98 vs ~0.89):
the defaults are hidden 100 / 30, dropout 0.7, 200 epochs (chosen on the 10 % validation split).

Usage:
    from src.gcmc import fit_gcmc, fit_gcmc_early
    from src.side_info import item_features
    F = item_features("data/namesngenre.npy")
    predict = fit_gcmc(train, shape, F, epochs=200)
    predict, curves = fit_gcmc_early(train, shape, F, valid, epochs=1000)   # valid RMSE every 10 epochs
    r_hat = predict(u, i)
"""
import numpy as np
import torch
from torch import nn

from src.data import clip, rmse

SEED = 0
LEVELS = np.arange(1, 11) / 2.0
LR = 0.01
SIDE_HIDDEN = 10
N_BASIS = 2


def level_index(r):
    """Index 0..9 of the rating levels 0.5..5."""
    return np.round(r * 2).astype(np.int64) - 1


def adjacency(train, shape):
    """List of 10 sparse users x items matrices with 1/sqrt(|N(u)| |N(i)|) on the edges of each level."""
    u, i, r = train
    nu = np.bincount(u, minlength=shape[0])
    ni = np.bincount(i, minlength=shape[1])
    w = 1.0 / np.sqrt(nu[u] * ni[i])
    lv = level_index(r)
    return [torch.sparse_coo_tensor(np.vstack([u[lv == k], i[lv == k]]), w[lv == k], shape,
                                    dtype=torch.float32).coalesce() for k in range(len(LEVELS))]


class GCMC(nn.Module):
    """One-layer GC-MC encoder (stack accumulation, ordinal sharing) and bilinear softmax decoder."""

    def __init__(self, shape, n_side, hidden=100, out=30):
        """Per-level embedding increments, dense layers, side-feature layer and decoder bases."""
        super().__init__()
        h = hidden // len(LEVELS)
        self.Tu = nn.Parameter(nn.init.xavier_uniform_(torch.empty(len(LEVELS), shape[0], h)))
        self.Ti = nn.Parameter(nn.init.xavier_uniform_(torch.empty(len(LEVELS), shape[1], h)))
        self.dense_u = nn.Linear(h * len(LEVELS), out, bias=False)
        self.dense_i = nn.Linear(h * len(LEVELS), out, bias=False)
        self.side = nn.Linear(n_side, SIDE_HIDDEN)
        self.side_i = nn.Linear(SIDE_HIDDEN, out, bias=False)
        self.P = nn.Parameter(nn.init.orthogonal_(torch.empty(N_BASIS, out, out)))
        self.a = nn.Parameter(nn.init.xavier_uniform_(torch.empty(len(LEVELS), N_BASIS)))

    def encode(self, A, F, dropout):
        """Final user and item representations (n_users, out), (n_items, out)."""
        Eu, Ei = torch.cumsum(self.Tu, 0), torch.cumsum(self.Ti, 0)
        Eu, Ei = node_dropout(Eu, dropout, self.training), node_dropout(Ei, dropout, self.training)
        hu = torch.relu(torch.cat([torch.sparse.mm(A[k], Ei[k]) for k in range(len(A))], 1))
        hi = torch.relu(torch.cat([torch.sparse.mm(A[k].t(), Eu[k]) for k in range(len(A))], 1))
        return self.dense_u(hu), self.dense_i(hi) + self.side_i(torch.relu(self.side(F)))

    def logits(self, zu, zi):
        """(n, 10) level logits of the aligned user / item representation rows."""
        basis = torch.stack([((zu @ self.P[b]) * zi).sum(1) for b in range(N_BASIS)], 1)
        return basis @ self.a.T


def node_dropout(E, p, training):
    """Drop whole nodes (rows of every level) with probability p, rescaling the kept ones."""
    if not training or p == 0:
        return E
    keep = (torch.rand(E.shape[1], 1) > p).float() / (1 - p)
    return E * keep[None]


def expected_rating(logits):
    """sum_r r softmax(logits)_r."""
    return torch.softmax(logits, -1) @ torch.as_tensor(LEVELS, dtype=torch.float32)


def setup(train, shape, F):
    """Model, adjacency, side features and training targets of one fit."""
    torch.manual_seed(SEED)
    model = GCMC(shape, F.shape[1])
    data = dict(A=adjacency(train, shape), F=torch.as_tensor(F, dtype=torch.float32),
                u=torch.as_tensor(train[0]), i=torch.as_tensor(train[1]),
                y=torch.as_tensor(level_index(train[2])))
    return model, data


def train_step(model, opt, data, dropout):
    """One full-batch Adam step on the level cross-entropy."""
    model.train()
    zu, zi = model.encode(data["A"], data["F"], dropout)
    loss = nn.functional.cross_entropy(model.logits(zu[data["u"]], zi[data["i"]]), data["y"])
    opt.zero_grad()
    loss.backward()
    opt.step()


@torch.no_grad()
def predict_fn(model, data):
    """predict(u, i) closure of the model in its current state (chunked expected ratings)."""
    model.eval()
    zu, zi = model.encode(data["A"], data["F"], 0.0)

    @torch.no_grad()
    def predict(u, i):
        out = np.empty(len(u))
        for s in range(0, len(u), 500_000):
            uu, ii = torch.as_tensor(u[s:s + 500_000]), torch.as_tensor(i[s:s + 500_000])
            out[s:s + 500_000] = expected_rating(model.logits(zu[uu], zi[ii])).numpy()
        return out

    return predict


def fit_gcmc(train, shape, F, epochs=200, dropout=0.7):
    """Train GC-MC for a fixed number of full-batch epochs and return predict(u, i)."""
    model, data = setup(train, shape, F)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(epochs):
        train_step(model, opt, data, dropout)
    return predict_fn(model, data)


def fit_gcmc_early(train, shape, F, valid, epochs=1000, dropout=0.7):
    """Train for `epochs`, recording the clipped valid RMSE every 10 epochs.

    Returns (predict at the last epoch, curves) with curves a list of (epoch, valid_rmse);
    used to choose the number of epochs on a validation split.
    """
    model, data = setup(train, shape, F)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    curves = []
    for e in range(epochs):
        train_step(model, opt, data, dropout)
        if (e + 1) % 10 == 0:
            p = predict_fn(model, data)(valid[0], valid[1])
            curves.append((e + 1, rmse(clip(p), valid[2])))
            print(curves[-1], flush=True)
    return predict_fn(model, data), curves
