"""Bayesian Factorization Machine (Rendle 2013), échantillonnage de Gibbs avec `myfm`.

Une note (u, i) est décrite par un vecteur de features x = [bloc user | bloc film] :
    bloc user : one-hot de u ; avec "iu", les films notés par u, chacun valant |N(u)|^-1/2
                (feedback implicite, comme SVD++) ;
    bloc film : one-hot de i ; avec "ii", les users ayant noté i, chacun valant |N(i)|^-1/2
                (feedback implicite « inversé ») ; avec "side", genres (multi-hot) et
                décennie de sortie (one-hot) du film, lus dans namesngenre2.npy.
Modèle FM :  r̂(x) = w0 + sum_j w_j x_j + sum_{j<l} <v_j, v_l> x_j x_l
Sans "iu"/"ii"/"side", c'est exactement BPMF avec biais (w0 + b_u + b_i + <v_u, v_i>).

Bayésien comme BPMF : chaque groupe de features a son prior gaussien avec hyperprior
(Normal-Gamma par groupe dans myfm). La précision du bruit alpha est tirée (prior Gamma,
alpha=None) ou fixée (alpha=valeur, via un prior Gamma très concentré). Les `burn_in` premières itérations sont jetées, la
prédiction est la moyenne des prédictions des échantillons suivants.

Les ensembles implicites N(u), N(i) ne sont construits qu'à partir des notes d'entraînement
(pas de fuite des positions de validation).

Modèle itératif (une époque = une itération de Gibbs) : si R_val est fourni, l'historique
contient à chaque itération la RMSE de validation de l'échantillon courant, celle de la
moyenne des échantillons après burn-in, et la précision du bruit alpha tirée.
"""
import re
import time
from functools import partial

import myfm
import myfm.base
import numpy as np
import scipy.sparse as sps
from tqdm import tqdm

from src.data import DATA_DIR
from src.models.base import Model, clip_ratings

GENRES_PATH = DATA_DIR / "namesngenre2.npy"   # fichier corrigé (mail du 19/10)
CHUNK = 100   # échantillons traités ensemble pour la moyenne dense
myfm.base.tqdm = partial(tqdm, disable=True)   # myfm affiche toujours une barre de progression


class BayesianFM(Model):
    iterative = True

    def __init__(self, rank=10, n_epochs=1000, burn_in=50, features=("iu", "ii", "side"),
                 alpha=None, genres_path=GENRES_PATH, seed=0, clip=True):
        self.rank, self.n_epochs, self.burn_in, self.alpha = rank, n_epochs, burn_in, alpha
        self.features, self.genres_path, self.seed, self.clip = tuple(features), genres_path, seed, clip

    @staticmethod
    def param_grid():
        return {"rank": [5, 10, 20]}

    def fit(self, R, R_val=None):
        users, items = np.nonzero(~np.isnan(R))
        r = R[users, items]
        n_users, n_items = R.shape

        user_groups = [sps.identity(n_users, format="csr")]
        item_groups = [sps.identity(n_items, format="csr")]
        if "iu" in self.features:
            user_groups.append(_implicit(users, items, (n_users, n_items)))
        if "ii" in self.features:
            item_groups.append(_implicit(items, users, (n_items, n_users)))
        if "side" in self.features:
            item_groups.append(sps.csr_matrix(item_side_features(self.genres_path)))
        Xu, Xi = sps.hstack(user_groups, format="csr"), sps.hstack(item_groups, format="csr")

        self.history = {"epoch": [], "val_rmse": [], "val_rmse_sample": [], "alpha": [], "time": []}
        t0 = time.perf_counter()
        val = None
        if R_val is not None:
            vu, vi = np.nonzero(~np.isnan(R_val))
            val = (vu, vi, R_val[vu, vi], np.zeros(len(vu)))   # dernier : somme des prédictions

        def callback(it, fm, hyper, history):
            """Appelé par myfm après chaque itération de Gibbs, avec l'échantillon courant."""
            self.history["epoch"].append(it + 1)
            self.history["alpha"].append(float(hyper.alpha))
            if val is not None:
                vu, vi, vr, vsum = val
                pred = _sample_entries(fm, Xu, Xi, vu, vi)
                self.history["val_rmse_sample"].append(_rmse(vr, pred, self.clip))
                if it >= self.burn_in:
                    vsum += pred
                    pred = vsum / (it + 1 - self.burn_in)
                self.history["val_rmse"].append(_rmse(vr, pred, self.clip))
            self.history["time"].append(time.perf_counter() - t0)
            return False, None

        model = myfm.MyFMRegressor(rank=self.rank, random_seed=self.seed, **_alpha_prior(self.alpha))
        model.fit(None, r, X_rel=[myfm.RelationBlock(users, Xu), myfm.RelationBlock(items, Xi)],
                  n_iter=self.n_epochs, n_kept_samples=self.n_epochs - self.burn_in,
                  group_shapes=[g.shape[1] for g in user_groups + item_groups], callback=callback)
        self._R_hat = _posterior_mean(model.predictor_.samples, Xu, Xi)
        return self

    def predict(self):
        return clip_ratings(self._R_hat) if self.clip else self._R_hat


def _alpha_prior(alpha):
    """Prior de myfm sur la précision du bruit : alpha ~ Gamma(alpha_0 / 2, taux beta_0 / 2).
    alpha=None : prior par défaut de myfm (alpha tiré, peu informatif). alpha=valeur : prior
    extrêmement concentré sur cette valeur (moyenne alpha_0 / beta_0, poids 1e8 pseudo-notes
    face à ~30 000 notes), ce qui revient à fixer alpha."""
    if alpha is None:
        return {}
    weight = 2e8
    return dict(alpha_0=weight, beta_0=weight / alpha)


def item_side_features(path=GENRES_PATH):
    """Genres (multi-hot) + décennie de sortie (one-hot, tirée du titre « Titre (Année) »)."""
    meta = np.load(path, allow_pickle=False)
    titles, genres = meta[:, 0], meta[:, 1]
    split = [g.split("|") for g in genres]
    genre_names = sorted({g for gs in split for g in gs})
    G = np.array([[name in gs for name in genre_names] for gs in split], dtype=float)
    years = [re.search(r"\((\d{4})\)\s*$", t) for t in titles]
    decades = [int(m.group(1)) // 10 * 10 if m else -1 for m in years]
    decade_names = sorted(set(decades))
    D = np.eye(len(decade_names))[[decade_names.index(d) for d in decades]]
    return np.hstack([G, D])


def _implicit(rows, cols, shape):
    """Matrice creuse (shape) valant |N(row)|^-1/2 à chaque position observée (row, col)."""
    n = np.bincount(rows, minlength=shape[0])
    return sps.csr_matrix((1.0 / np.sqrt(n[rows]), (rows, cols)), shape=shape)


def _posterior_mean(samples, Xu, Xi):
    """Moyenne, sur les échantillons gardés, de la matrice dense des prédictions.

    Toutes les features sont soit côté user, soit côté film, donc pour un échantillon
    (w0, w, V) la FM se sépare en  w0 + g_u + g_i + <a_u, a_i>  avec, pour un bloc X :
        a = X V,   g = X w + (|X V|^2 - X^2 V^2) / 2   (interactions internes au bloc).
    """
    p = Xu.shape[1]
    total = np.zeros((Xu.shape[0], Xi.shape[0]))
    for start in range(0, len(samples), CHUNK):
        a_users, a_items = [], []
        for fm in samples[start:start + CHUNK]:
            g_u, a_u = _block_terms(Xu, fm.w[:p], fm.V[:p])
            g_i, a_i = _block_terms(Xi, fm.w[p:], fm.V[p:])
            total += fm.w0 + g_u[:, None] + g_i[None, :]
            a_users.append(a_u)
            a_items.append(a_i)
        total += np.hstack(a_users) @ np.hstack(a_items).T    # somme des <a_u, a_i> du paquet
    return total / len(samples)


def _sample_entries(fm, Xu, Xi, us, its):
    """Prédictions d'un échantillon aux positions (us, its)."""
    p = Xu.shape[1]
    g_u, a_u = _block_terms(Xu, fm.w[:p], fm.V[:p])
    g_i, a_i = _block_terms(Xi, fm.w[p:], fm.V[p:])
    return fm.w0 + g_u[us] + g_i[its] + np.einsum("nk,nk->n", a_u[us], a_i[its])


def _rmse(truth, pred, clip):
    pred = clip_ratings(pred) if clip else pred
    return float(np.sqrt(np.mean((truth - pred) ** 2)))


def _block_terms(X, w, V):
    a = X @ V
    g = X @ w + 0.5 * (np.sum(a ** 2, axis=1) - X.multiply(X) @ np.sum(V ** 2, axis=1))
    return g, a
