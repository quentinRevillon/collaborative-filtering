"""Modèle de voisinage « global » de Koren (KDD 2008, « Factorization Meets the
Neighborhood », sans le terme de feedback implicite c_ij) :

    r̂_ui = mu + b_u + b_i + |R^k(i;u)|^(-1/2) * sum_{j in R^k(i;u)} w_ij (r_uj - b̃_uj)

- b̃_uj : baseline FIXÉE (modèle Biases appris avant), donc e_uj = r_uj - b̃_uj est fixé ;
- S^k(i) : les k films les plus similaires à i (Pearson sur les résidus + shrink, comme
  KNNResidual) ; R^k(i;u) = films notés par u parmi S^k(i) ;
- w_ij, b_u, b_i : paramètres APPRIS (contrairement au kNN où le poids de j vient de la
  similarité), avec la fonction de coût
      C = sum_{(u,i) in S} (r_ui - r̂_ui)^2 + lam_u ||b_u||^2 + lam_i ||b_i||^2 + lam_w ||W||^2

À e fixé, r̂ est linéaire en theta = (b_u, b_i, W) : C est une régression ridge, convexe.
On la résout par gradient conjugué sur les équations normales (X^T X + Lambda) theta = X^T y,
sans pas d'apprentissage ; une époque = une itération de gradient conjugué. On part des
biais de la baseline et de W = 0 (époque 0 = la baseline).
"""
import time

import numpy as np

from src.models.base import Model, clip_ratings
from src.models.baselines import Biases
from src.models.knn import _pearson_on_columns


class KorenNeighborhood(Model):
    iterative = True

    def __init__(self, k=50, lam_w=10.0, lam_u=5.0, lam_i=3.0, shrink=100.0, n_epochs=50, clip=True):
        self.k, self.lam_w, self.lam_u, self.lam_i = k, lam_w, lam_u, lam_i
        self.shrink, self.n_epochs, self.clip = shrink, n_epochs, clip

    @staticmethod
    def param_grid():
        return {"k": [20, 50, 100], "lam_w": [0.1, 1, 10, 100, 1000]}

    def fit(self, R, R_val=None):
        mask = ~np.isnan(R)
        users, items = np.nonzero(mask)
        r = R[users, items]
        n_users, n_items = R.shape

        base = Biases(lam_u=self.lam_u, lam_i=self.lam_i, clip=False).fit(R)
        self.mu, self._mask = base.mu, mask
        self._E = np.where(mask, R - base.predict(), 0.0)

        # Voisinage S^k(i) : les k films les plus similaires (Pearson sur résidus, shrink)
        rho, n = _pearson_on_columns(self._E.astype(np.float32), mask)
        s = rho * np.divide(n, n + self.shrink, out=np.zeros_like(n), where=(n + self.shrink) > 0)
        np.fill_diagonal(s, -np.inf)
        self.neighbors = np.argpartition(-s, self.k - 1, axis=1)[:, :self.k]   # (n_items, k)

        # theta = [b_u | b_i | W.ravel()], et sa pénalité ridge
        sizes = (n_users, n_items, n_items * self.k)
        lam = np.concatenate([np.full(sizes[0], float(self.lam_u)), np.full(sizes[1], float(self.lam_i)),
                              np.full(sizes[2], float(self.lam_w))])
        X = _Design(users, items, *self._pairs(users, items), sizes)
        y = r - self.mu
        theta = np.concatenate([base.b_u, base.b_i, np.zeros(sizes[2])])
        val = None
        if R_val is not None:
            vu, vi = np.nonzero(~np.isnan(R_val))
            val = (_Design(vu, vi, *self._pairs(vu, vi), sizes), R_val[vu, vi])

        # Gradient conjugué sur (X^T X + Lambda) theta = X^T y
        def A(v):
            return X.rmatvec(X.matvec(v)) + lam * v

        res = X.rmatvec(y) - A(theta)
        p = res.copy()
        rs = res @ res
        self.history = {"epoch": [], "loss": [], "train_rmse": [], "val_rmse": [], "time": []}
        t0 = time.perf_counter()
        for epoch in range(1, self.n_epochs + 1):
            Ap = A(p)
            alpha = rs / (p @ Ap)
            theta += alpha * p
            res -= alpha * Ap
            rs_new = res @ res
            p = res + (rs_new / rs) * p
            rs = rs_new
            self._set_params(theta, sizes)
            self._log(epoch, X, y, theta, lam, val, t0)
        return self

    def _pairs(self, us, its):
        """Pour chaque note (us[n], its[n]) : les voisins j de its[n] notés par us[n].
        Renvoie (indice de la note, indice de w_ij dans W.ravel(), valeur |R^k|^-1/2 e_uj)."""
        cand = self.neighbors[its]                                  # (N, k)
        obs = self._mask[us[:, None], cand]                         # j noté par u ?
        count = obs.sum(axis=1)
        norm = np.divide(1.0, np.sqrt(count), out=np.zeros(len(us)), where=count > 0)
        rows, slots = np.nonzero(obs)
        cols = its[rows] * self.k + slots
        vals = norm[rows] * self._E[us[rows], cand[rows, slots]]
        return rows, cols, vals

    def _set_params(self, theta, sizes):
        self.b_u = theta[:sizes[0]]
        self.b_i = theta[sizes[0]:sizes[0] + sizes[1]]
        self.W = theta[sizes[0] + sizes[1]:].reshape(-1, self.k)

    def _log(self, epoch, X, y, theta, lam, val, t0):
        err = y - X.matvec(theta)
        sq = float(err @ err)
        self.history["epoch"].append(epoch)
        self.history["loss"].append(sq + float(lam @ theta ** 2))
        self.history["train_rmse"].append(np.sqrt(sq / len(y)))
        if val is not None:
            Xv, rv = val
            pred = clip_ratings(self.mu + Xv.matvec(theta)) if self.clip else self.mu + Xv.matvec(theta)
            self.history["val_rmse"].append(float(np.sqrt(np.mean((rv - pred) ** 2))))
        self.history["time"].append(time.perf_counter() - t0)

    def predict(self):
        n_users, n_items = self._mask.shape
        P = self.mu + self.b_u[:, None] + self.b_i[None, :]
        for start in range(0, n_items, 256):                       # par paquets de films
            its = np.arange(start, min(start + 256, n_items))
            nb = self.neighbors[its]                                # (c, k)
            E_nb = self._E[:, nb]                                   # (n_users, c, k), 0 si non noté
            count = self._mask[:, nb].sum(axis=2)
            norm = np.divide(1.0, np.sqrt(count), out=np.zeros(count.shape), where=count > 0)
            P[:, its] += norm * np.einsum("ucm,cm->uc", E_nb, self.W[its])
        return clip_ratings(P) if self.clip else P


class _Design:
    """Matrice de design creuse X (une ligne par note) : X theta = b_u + b_i + sum w_ij x_ij."""

    def __init__(self, users, items, rows, cols, vals, sizes):
        self.users, self.items, self.rows, self.cols, self.vals = users, items, rows, cols, vals
        self.sizes, self.n = sizes, len(users)

    def matvec(self, theta):
        nu, ni, _ = self.sizes
        W = theta[nu + ni:]
        return (theta[self.users] + theta[nu + self.items]
                + np.bincount(self.rows, weights=self.vals * W[self.cols], minlength=self.n))

    def rmatvec(self, v):
        nu, ni, nw = self.sizes
        return np.concatenate([np.bincount(self.users, weights=v, minlength=nu),
                               np.bincount(self.items, weights=v, minlength=ni),
                               np.bincount(self.cols, weights=self.vals * v[self.rows], minlength=nw)])
