"""kNN sur les résidus du modèle de biais (Koren 2008, « Factorization meets the neighborhood »).

    r̂_ui = b_ui + sum_{j in N_k(i; u)} s_ij e_uj / (sum_{j in N_k(i; u)} |s_ij| + damp)

avec b_ui = mu + b_u + b_i (modèle Biases) et e_uj = r_uj - b_uj ses résidus.
Version films (kind="item") : N_k(i; u) = les k films notés par u les plus similaires à i.
Version users (kind="user") : symétrique, les k users ayant noté i les plus similaires à u.

Similarité : corrélation de Pearson calculée sur les résidus (« pearson baseline ») et
sur les seuls users (resp. films) en commun, puis rétrécie selon leur nombre n_ij :
    rho_ij = sum_u e_ui e_uj / sqrt(sum_u e_ui^2 * sum_u e_uj^2)     (u ∈ notés par i et j)
    s_ij   = n_ij / (n_ij + shrink) * rho_ij
Le shrink fait tendre vers 0 une corrélation estimée sur trop peu de notes communes.

k, shrink et damp n'interviennent qu'à la prédiction : on peut les changer après fit()
sans recalculer les corrélations (utile pour balayer une grille rapidement).
"""
import numpy as np

from src.models.base import Model, clip_ratings
from src.models.baselines import Biases


class KNNResidual(Model):
    def __init__(self, kind="item", k=40, shrink=100.0, damp=0.0, lam_u=5.0, lam_i=3.0, clip=True):
        self.kind, self.k, self.shrink, self.damp = kind, k, shrink, damp
        self.lam_u, self.lam_i, self.clip = lam_u, lam_i, clip

    @staticmethod
    def param_grid():
        return {"k": [10, 20, 40, 80], "shrink": [0, 10, 50, 100]}

    def fit(self, R, R_val=None):
        self.baseline = Biases(lam_u=self.lam_u, lam_i=self.lam_i, clip=False).fit(R)
        B = self.baseline.predict()
        mask = ~np.isnan(R)
        E = np.where(mask, R - B, 0.0)
        # On se ramène toujours au cas « voisins = colonnes » : user-kNN = item-kNN sur R^T
        if self.kind == "user":
            E, mask, B = E.T, mask.T, B.T
        self._B, self._E, self._mask = B, E.astype(np.float32), mask
        self._rho, self._n = _pearson_on_columns(self._E, mask)
        return self

    def similarity(self):
        denom = self._n + self.shrink
        s = self._rho * np.divide(self._n, denom, out=np.zeros_like(self._n), where=denom > 0)
        np.fill_diagonal(s, 0.0)
        return s

    def predict(self):
        s = self.similarity()
        P = self._B.copy()
        for a in range(P.shape[0]):
            rated = np.flatnonzero(self._mask[a])
            if len(rated) == 0:
                continue
            S = s[:, rated]                                  # (n_cols, n_rated)
            e = self._E[a, rated]
            if len(rated) > self.k:                          # garder les k plus similaires
                top = np.argpartition(-S, self.k - 1, axis=1)[:, :self.k]
                S, e = np.take_along_axis(S, top, axis=1), e[top]
            num = (S * e).sum(axis=1)
            den = np.abs(S).sum(axis=1) + self.damp
            P[a] += np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        if self.kind == "user":
            P = P.T
        return clip_ratings(P) if self.clip else P

    def val_rmse_grid(self, R_val, ks, damps):
        """RMSE sur les notes de R_val pour toutes les combinaisons (k, damp), au shrink
        courant, sans calculer la matrice complète : on ne prédit que les notes de R_val,
        et les voisins de chaque note sont triés une seule fois pour tous les k."""
        val = ~np.isnan(R_val)
        target = R_val.T if self.kind == "user" else R_val
        val = val.T if self.kind == "user" else val
        s = self.similarity()
        k_max = max(ks)
        sq_err = {(k, d): 0.0 for k in ks for d in damps}
        count = 0
        for a in range(val.shape[0]):
            cols = np.flatnonzero(val[a])
            rated = np.flatnonzero(self._mask[a])
            if len(cols) == 0:
                continue
            base, truth = self._B[a, cols], target[a, cols]
            count += len(cols)
            if len(rated) == 0:
                for key in sq_err:
                    sq_err[key] += float(((truth - clip_ratings(base)) ** 2).sum())
                continue
            S = s[np.ix_(cols, rated)]
            e = self._E[a, rated]
            order = np.argsort(-S, axis=1)[:, :k_max]       # voisins du plus au moins similaire
            S = np.take_along_axis(S, order, axis=1)
            num_cum = np.cumsum(S * e[order], axis=1)
            den_cum = np.cumsum(np.abs(S), axis=1)
            for k in ks:
                kk = min(k, S.shape[1]) - 1
                for d in damps:
                    den = den_cum[:, kk] + d
                    corr = np.divide(num_cum[:, kk], den, out=np.zeros(len(cols)), where=den > 0)
                    sq_err[(k, d)] += float(((truth - clip_ratings(base + corr)) ** 2).sum())
        return {key: np.sqrt(v / count) for key, v in sq_err.items()}


def _pearson_on_columns(E, mask):
    """Corrélation de Pearson entre colonnes de E sur les lignes communes, et leur nombre."""
    M = mask.astype(np.float32)
    n = M.T @ M                                  # n_ij : nb de lignes où i et j sont observés
    num = E.T @ E                                # sum e_i e_j (E vaut 0 hors observations)
    sq = (E ** 2).T @ M                          # sq_ij = sum_{lignes communes} e_i^2
    den = np.sqrt(sq * sq.T)
    rho = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return rho, n
