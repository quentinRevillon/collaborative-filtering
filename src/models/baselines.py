"""Baselines : moyenne globale et modèle de biais  r_ui ≈ mu + b_u + b_i."""
import numpy as np

from src.models.base import Model, clip_ratings


class GlobalMean(Model):
    """Prédit la même valeur mu (moyenne des notes observées) partout."""

    def fit(self, R, R_val=None):
        self.shape = R.shape
        self.mu = np.nanmean(R)
        return self

    def predict(self):
        return np.full(self.shape, self.mu)


class Biases(Model):
    """r_ui ≈ mu + b_u + b_i, appris par moindres carrés sur les notes observées.

    Minimise  sum_{(u,i) obs} (r_ui - mu - b_u - b_i)^2 + lam_u ||b_u||^2 + lam_i ||b_i||^2
    par minimisation alternée exacte (chaque étape est un problème ridge 1D en
    forme close) :
        b_i = sum_u (r_ui - mu - b_u) / (lam_i + n_i)
        b_u = sum_i (r_ui - mu - b_i) / (lam_u + n_u)

    - lam_u = lam_i = 0 : biais non régularisés (moyenne des résidus).
    - lam > 0 : les biais des users/films avec peu de notes sont tirés vers 0
      (shrinkage), un film noté 1 fois ne reçoit plus un biais extrême.

    Options pour reproduire les baselines simples :
    - use_user=False -> mu + b_i (moyenne par film, éventuellement régularisée)
    - use_item=False -> mu + b_u (moyenne par user)
    - n_iter=1 -> estimation séquentielle classique (b_i d'abord, puis b_u sur les résidus)
    """

    def __init__(self, lam_u=0.0, lam_i=0.0, n_iter=10, use_user=True, use_item=True, clip=True):
        self.lam_u, self.lam_i = lam_u, lam_i
        self.n_iter = n_iter
        self.use_user, self.use_item = use_user, use_item
        self.clip = clip

    @staticmethod
    def param_grid():
        return {"lam_u": [0, 1, 2, 3, 5, 10, 20], "lam_i": [0, 1, 2, 3, 5, 10, 20]}

    def fit(self, R, R_val=None):
        users, items = np.nonzero(~np.isnan(R))
        r = R[users, items]
        n_users, n_items = R.shape
        n_u = np.bincount(users, minlength=n_users)
        n_i = np.bincount(items, minlength=n_items)

        self.mu = r.mean()
        self.b_u, self.b_i = np.zeros(n_users), np.zeros(n_items)

        for _ in range(self.n_iter):
            if self.use_item:
                resid = r - self.mu - self.b_u[users]
                self.b_i = _shrunk_mean(np.bincount(items, resid, n_items), n_i, self.lam_i)
            if self.use_user:
                resid = r - self.mu - self.b_i[items]
                self.b_u = _shrunk_mean(np.bincount(users, resid, n_users), n_u, self.lam_u)
        return self

    def predict(self):
        R_hat = self.mu + self.b_u[:, None] + self.b_i[None, :]
        return clip_ratings(R_hat) if self.clip else R_hat


def _shrunk_mean(sums, counts, lam):
    """sums / (lam + counts), avec 0 quand il n'y a aucune observation (et lam = 0)."""
    denom = lam + counts
    return np.divide(sums, denom, out=np.zeros_like(sums, dtype=float), where=denom > 0)
