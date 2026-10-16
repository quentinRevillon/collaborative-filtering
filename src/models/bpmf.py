"""Bayesian Probabilistic Matrix Factorization (Salakhutdinov & Mnih, ICML 2008),
échantillonnage de Gibbs.

Modèle (notes centrées par mu, comme dans le code des auteurs) :
    r_ui | U, V      ~ N(mu + <U_u, V_i>, 1/alpha)                 vraisemblance
    U_u  | mu_U, L_U ~ N(mu_U, L_U^-1)        V_i | mu_V, L_V ~ N(mu_V, L_V^-1)   prior
    L_U ~ Wishart(W0, nu0),  mu_U | L_U ~ N(mu0, (beta0 L_U)^-1)   (idem pour V)   hyperprior
Valeurs du papier : alpha = 2, beta0 = 2, nu0 = D, W0 = Id, mu0 = 0. Ici alpha = 1.5 par défaut :
optimum de la CV 5 plis sur nos données (src/experiments/bpmf_alpha.py).

Variante bias="full" (celle des slides) : on ajoute le biais dans le vecteur,
w_u = (U_u, b_u) et w_i = (V_i, b_i), r̂_ui = mu + <U_u, V_i> + b_u + b_i ; le prior
gaussien et l'hyperprior portent alors sur des vecteurs de dimension D + 1.

Une itération de Gibbs (une « époque ») :
    1. (mu_U, L_U) ~ p(. | U)            loi Gauss-Wishart a posteriori (conjuguée)
    2. chaque U_u  ~ N(m_u, A_u^-1)      A_u = L_U + alpha sum_{i notés} x_i x_i^T
                                         m_u = A_u^-1 (L_U mu_U + alpha sum_i y_ui x_i)
    3. idem pour (mu_V, L_V) puis chaque V_i, avec les U qui viennent d'être tirés.
Prédiction : moyenne des prédictions des échantillons tirés après les `burn_in`
premières itérations (approximation de l'espérance a posteriori).
alpha="gibbs" : alpha n'est plus fixé mais tiré à chaque itération (prior conjugué Gamma) :
    alpha ~ Gamma(a0, b0)                                            prior (forme, taux)
    alpha | R, U, V ~ Gamma(a0 + |S|/2,  b0 + (1/2) sum_{(u,i) in S} (r_ui - r̂_ui)^2)
(étape 4 de chaque itération, avec les U, V qui viennent d'être tirés ; départ à alpha_init).

Initialisation : `init_epochs` itérations d'ALS (le papier part de la solution MAP de PMF) ;
init_epochs = 0 : départ aléatoire, U, V ~ N(0, 0.1^2), biais nuls.
"""
import time

import numpy as np

from src.models.base import Model, clip_ratings
from src.models.mf import MatrixFactorizationALS


class BPMF(Model):
    iterative = True

    def __init__(self, rank=10, alpha=1.5, bias="mean", n_epochs=100, burn_in=20, beta0=2.0,
                 init_epochs=10, init_lam=10.0, a0=1.0, b0=1.0, alpha_init=2.0, seed=0, clip=True):
        self.rank, self.alpha, self.bias, self.n_epochs, self.burn_in = rank, alpha, bias, n_epochs, burn_in
        self.a0, self.b0, self.alpha_init = a0, b0, alpha_init
        self.beta0, self.init_epochs, self.init_lam = beta0, init_epochs, init_lam
        self.seed, self.clip = seed, clip

    def fit(self, R, R_val=None):
        rng = np.random.default_rng(self.seed)
        mask = ~np.isnan(R)
        users, items = np.nonzero(mask)
        r = R[users, items]
        n_users, n_items = R.shape
        D, full = self.rank, self.bias == "full"
        self.mu = r.mean()

        # Point de départ de la chaîne : quelques itérations d'ALS (solution MAP-like), ou aléatoire
        if self.init_epochs > 0:
            init = MatrixFactorizationALS(rank=D, lam=self.init_lam, bias=self.bias,
                                          n_epochs=self.init_epochs, seed=self.seed).fit(R)
            U, V, b_u, b_i = init.U, init.I, init.b_u, init.b_i
        else:
            U, V = rng.normal(0, 0.1, (n_users, D)), rng.normal(0, 0.1, (n_items, D))
            b_u, b_i = np.zeros(n_users), np.zeros(n_items)
        A = np.hstack([U, b_u[:, None]]) if full else U.copy()   # côté users
        B = np.hstack([V, b_i[:, None]]) if full else V.copy()   # côté films
        d = A.shape[1]
        hyper = dict(mu0=np.zeros(d), W0_inv=np.eye(d), nu0=d, beta0=self.beta0)

        val = None
        if R_val is not None:
            vu, vi = np.nonzero(~np.isnan(R_val))
            val = (vu, vi, R_val[vu, vi], np.zeros(len(vu)))   # dernier : somme des prédictions
        self._sum, self.n_samples = np.zeros(R.shape), 0
        sample_alpha = self.alpha == "gibbs"
        alpha = self.alpha_init if sample_alpha else float(self.alpha)
        self.history = {"epoch": [], "loss": [], "train_rmse": [], "val_rmse": [],
                        "val_rmse_sample": [], "alpha": [], "time": []}
        t0 = time.perf_counter()
        for epoch in range(1, self.n_epochs + 1):
            # Users : features x_i = V_i (+ 1), cible y = r - mu (- b_i)
            mu_h, lam_h = _sample_gauss_wishart(A, rng, **hyper)
            X, y = self._features(B, items, r)
            A = _sample_rows(users, X, y, n_users, mu_h, lam_h, alpha, rng)
            # Films : features x_u = U_u (+ 1), cible y = r - mu (- b_u)
            mu_h, lam_h = _sample_gauss_wishart(B, rng, **hyper)
            X, y = self._features(A, users, r)
            B = _sample_rows(items, X, y, n_items, mu_h, lam_h, alpha, rng)
            # Bruit : alpha ~ Gamma(a0 + |S|/2, taux b0 + SSE/2)
            if sample_alpha:
                e = r - self._entries(A, B, users, items)
                alpha = rng.gamma(self.a0 + len(r) / 2, 1.0 / (self.b0 + 0.5 * float(e @ e)))

            self.A, self.B = A, B
            self.history["alpha"].append(float(alpha))
            if epoch > self.burn_in:
                self._sum += self._dense(A, B)
                self.n_samples += 1
            self._log(epoch, users, items, r, val, t0)
        return self

    def _features(self, other, idx, r):
        """Features et cibles pour tirer un côté, l'autre côté (`other`) étant fixé."""
        D = self.rank
        if self.bias == "full":
            X = np.hstack([other[idx, :D], np.ones((len(idx), 1))])
            return X, r - self.mu - other[idx, D]
        return other[idx], r - self.mu

    def _entries(self, A, B, us, its):
        D = self.rank
        pred = self.mu + np.einsum("nk,nk->n", A[us, :D], B[its, :D])
        if self.bias == "full":
            pred += A[us, D] + B[its, D]
        return pred

    def _dense(self, A, B):
        D = self.rank
        P = self.mu + A[:, :D] @ B[:, :D].T
        if self.bias == "full":
            P += A[:, D][:, None] + B[:, D][None, :]
        return P

    def _log(self, epoch, users, items, r, val, t0):
        e = r - self._entries(self.A, self.B, users, items)
        self.history["epoch"].append(epoch)
        self.history["loss"].append(float(e @ e))
        self.history["train_rmse"].append(float(np.sqrt(np.mean(e ** 2))))
        if val is not None:
            vu, vi, vr, vsum = val
            sample = self._entries(self.A, self.B, vu, vi)
            self.history["val_rmse_sample"].append(_rmse(vr, sample, self.clip))
            if epoch > self.burn_in:
                vsum += sample
                self.history["val_rmse"].append(_rmse(vr, vsum / self.n_samples, self.clip))
            else:  # pas encore de moyenne : on note l'échantillon courant
                self.history["val_rmse"].append(self.history["val_rmse_sample"][-1])
        self.history["time"].append(time.perf_counter() - t0)

    def predict(self):
        P = self._sum / self.n_samples if self.n_samples else self._dense(self.A, self.B)
        return clip_ratings(P) if self.clip else P


def _rmse(truth, pred, clip):
    pred = clip_ratings(pred) if clip else pred
    return float(np.sqrt(np.mean((truth - pred) ** 2)))


def _sample_gauss_wishart(X, rng, mu0, W0_inv, nu0, beta0):
    """Tire (mu, Lambda) selon la loi a posteriori Gauss-Wishart, sachant les lignes de X
    (formules de l'annexe du papier)."""
    N, d = X.shape
    x_bar = X.mean(axis=0)
    S = (X - x_bar).T @ (X - x_bar) / N
    beta_n, nu_n = beta0 + N, nu0 + N
    mu_n = (beta0 * mu0 + N * x_bar) / beta_n
    diff = (mu0 - x_bar)[:, None]
    W_n = np.linalg.inv(W0_inv + N * S + (beta0 * N / beta_n) * diff @ diff.T)
    lam = _sample_wishart((W_n + W_n.T) / 2, nu_n, rng)
    mu = mu_n + _sample_with_precision(beta_n * lam, rng)
    return mu, lam


def _sample_wishart(W, nu, rng):
    """Tirage Wishart(W, nu) par la décomposition de Bartlett."""
    d = W.shape[0]
    L = np.linalg.cholesky(W)
    T = np.tril(rng.standard_normal((d, d)), k=-1)
    T[np.diag_indices(d)] = np.sqrt(rng.chisquare(nu - np.arange(d)))
    LT = L @ T
    return LT @ LT.T


def _sample_with_precision(P, rng):
    """Vecteur gaussien centré de matrice de précision P (covariance P^-1)."""
    L = np.linalg.cholesky(P)
    return np.linalg.solve(L.T, rng.standard_normal(P.shape[0]))


def _sample_rows(rows, X, y, n_rows, mu_h, lam_h, alpha, rng):
    """Tire chaque ligne w_j ~ N(m_j, A_j^-1), toutes à la fois (calcul par lots) :
        A_j = lam_h + alpha sum_{n : rows[n]=j} x_n x_n^T
        m_j = A_j^-1 (lam_h mu_h + alpha sum_n y_n x_n)
    Une ligne sans observation est tirée selon le prior N(mu_h, lam_h^-1)."""
    d = X.shape[1]
    outer = (X[:, :, None] * X[:, None, :]).reshape(len(X), d * d)
    G = np.stack([np.bincount(rows, weights=outer[:, c], minlength=n_rows)
                  for c in range(d * d)], axis=1).reshape(n_rows, d, d)
    b = np.stack([np.bincount(rows, weights=X[:, c] * y, minlength=n_rows) for c in range(d)], axis=1)
    precision = lam_h[None] + alpha * G
    rhs = (lam_h @ mu_h)[None] + alpha * b
    mean = np.linalg.solve(precision, rhs[..., None])[..., 0]
    L = np.linalg.cholesky(precision)
    noise = np.linalg.solve(np.swapaxes(L, 1, 2), rng.standard_normal((n_rows, d))[..., None])[..., 0]
    return mean + noise
