"""Factorisation de matrice, notations du cours : R ≈ U I^T.

Ici R est (users x films), donc U : (n_users, k) et I : (n_films, k), et
r̂_ui = <u_u, i_i>. Trois variantes selon `bias` :
    "none" : r̂_ui = <u_u, i_i>                       (modèle pur du cours)
    "mean" : r̂_ui = mu + <u_u, i_i>                  (données centrées)
    "full" : r̂_ui = mu + b_u + b_i + <u_u, i_i>      (biais appris en même temps)

Fonction de coût (somme sur les notes observées S, comme dans le cours) :
    C = sum_{(u,i) in S} (r_ui - r̂_ui)^2
        + lam (||U||^2 + ||I||^2) + lam_bias (||b_u||^2 + ||b_i||^2)
Le cours a deux coefficients (lambda pour I, mu pour U) ; on prend le même `lam`.

Deux optimiseurs de ce même C : MatrixFactorizationGD et MatrixFactorizationALS.
"""
import time

import numpy as np

from src.baselines import RATING_MAX, RATING_MIN


class _MatrixFactorization:
    """Partie commune : paramètres, initialisation, prédiction, historique."""

    def __init__(self, rank=10, lam=1.0, lam_bias=2.0, n_epochs=200, bias="full",
                 init_std=0.1, seed=0, clip=True):
        self.rank, self.lam, self.lam_bias, self.n_epochs = rank, lam, lam_bias, n_epochs
        self.bias, self.init_std, self.seed, self.clip = bias, init_std, seed, clip

    def fit(self, R, R_val=None):
        """Entraîne sur R. Si R_val est fourni, enregistre la RMSE de validation à chaque
        époque dans self.history (utile pour les courbes d'apprentissage)."""
        users, items, r = _observed(R)
        val = _observed(R_val) if R_val is not None else None
        n_users, n_items = R.shape
        rng = np.random.default_rng(self.seed)

        self.mu = r.mean() if self.bias in ("mean", "full") else 0.0
        self.b_u, self.b_i = np.zeros(n_users), np.zeros(n_items)
        self.U = rng.normal(0, self.init_std, (n_users, self.rank))
        self.I = rng.normal(0, self.init_std, (n_items, self.rank))

        self.history = {"epoch": [], "loss": [], "train_rmse": [], "val_rmse": [], "time": []}
        t0 = time.perf_counter()
        self._prepare(users, items, n_users, n_items)
        for epoch in range(1, self.n_epochs + 1):
            self._step(users, items, r, n_users, n_items)
            self._log(epoch, users, items, r, val, t0)
        return self

    def _prepare(self, users, items, n_users, n_items):
        pass

    def _step(self, users, items, r, n_users, n_items):
        raise NotImplementedError

    def _log(self, epoch, users, items, r, val, t0):
        e = r - self._predict_entries(users, items)
        sq_err = float(e @ e)
        if not np.isfinite(sq_err) or sq_err > 1e12:
            raise FloatingPointError(f"divergence à l'époque {epoch}")
        loss = (sq_err + self.lam * ((self.U ** 2).sum() + (self.I ** 2).sum())
                + self.lam_bias * ((self.b_u ** 2).sum() + (self.b_i ** 2).sum()))
        self.history["epoch"].append(epoch)
        self.history["loss"].append(float(loss))
        self.history["train_rmse"].append(np.sqrt(sq_err / len(r)))
        if val is not None:
            val_users, val_items, val_r = val
            val_e = val_r - self._predict_entries(val_users, val_items, clip=self.clip)
            self.history["val_rmse"].append(float(np.sqrt(np.mean(val_e ** 2))))
        self.history["time"].append(time.perf_counter() - t0)

    def _predict_entries(self, users, items, clip=False):
        pred = (self.mu + self.b_u[users] + self.b_i[items]
                + np.einsum("nk,nk->n", self.U[users], self.I[items]))
        return np.clip(pred, RATING_MIN, RATING_MAX) if clip else pred

    def predict(self):
        R_hat = self.mu + self.b_u[:, None] + self.b_i[None, :] + self.U @ self.I.T
        return np.clip(R_hat, RATING_MIN, RATING_MAX) if self.clip else R_hat


class MatrixFactorizationGD(_MatrixFactorization):
    """Descente de gradient « full batch » : à chaque époque, gradient exact de C
    sur toutes les notes observées, puis mise à jour simultanée de U, I (et des biais).

    Avec e_ui = r_ui - r̂_ui sur les notes observées (cf. slide « Missing values ») :
        dC/du_u = -2 sum_{i notés par u} e_ui i_i + 2 lam u_u
        dC/di_i = -2 sum_{u ayant noté i} e_ui u_u + 2 lam i_i
        dC/db_u = -2 sum_i e_ui + 2 lam_bias b_u        (idem pour b_i)

    Les calculs ne portent que sur les ~30k notes observées (vecteurs users, items,
    ratings), pas sur la matrice dense 610 x 4980 : une époque coûte O(|S| k).

    momentum > 0 active la méthode du « heavy ball » : v <- momentum * v - lr * grad,
    theta <- theta + v. Même point fixe que la GD simple, mais convergence bien plus rapide.
    """

    def __init__(self, lr=1e-3, momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        self.lr, self.momentum = lr, momentum

    def _prepare(self, users, items, n_users, n_items):
        self._velocity = [np.zeros_like(p) for p in (self.U, self.I, self.b_u, self.b_i)]

    def _step(self, users, items, r, n_users, n_items):
        e = r - self._predict_entries(users, items)
        grads = [
            -2 * _scatter_sum(users, e[:, None] * self.I[items], n_users) + 2 * self.lam * self.U,
            -2 * _scatter_sum(items, e[:, None] * self.U[users], n_items) + 2 * self.lam * self.I,
        ]
        if self.bias == "full":
            grads += [-2 * _scatter_sum(users, e, n_users) + 2 * self.lam_bias * self.b_u,
                      -2 * _scatter_sum(items, e, n_items) + 2 * self.lam_bias * self.b_i]
        for p, v, g in zip((self.U, self.I, self.b_u, self.b_i), self._velocity, grads):
            v *= self.momentum
            v -= self.lr * g
            p += v  # mise à jour en place : self.U, self.I, ... sont modifiés


class MatrixFactorizationALS(_MatrixFactorization):
    """Alternating Least Squares : U fixé, on minimise C exactement en I, puis
    l'inverse (cf. slide 19 : I_{t+1} = f(U_t), puis U_{t+1} = f(I_{t+1})).

    Avec données manquantes, la forme close du cours I = R U (U^T U + lam Id)^-1
    devient un problème ridge indépendant par film, sur les seuls users qui l'ont noté :
        i_i = (sum_{u ∈ S_i} u_u u_u^T + lam Id)^-1  sum_{u ∈ S_i} y_ui u_u
    avec y_ui = r_ui - mu - b_u (la partie de la note que les facteurs doivent expliquer).
    Symétrique pour u_u.

    Variante "full" : on apprend b_i en même temps que i_i en ajoutant une colonne
    de 1 aux u_u : x_u = [u_u, 1], w_i = [i_i, b_i], pénalité diag(lam, ..., lam, lam_bias).

    Pas de pas d'apprentissage : chaque demi-étape est un minimum exact, donc C décroît
    à chaque itération. Une époque = une mise à jour de I puis une de U.
    """

    def _prepare(self, users, items, n_users, n_items):
        self._by_user = _groups(users, n_users)
        self._by_item = _groups(items, n_items)

    def _step(self, users, items, r, n_users, n_items):
        y_items = r - self.mu - (self.b_u[users] if self.bias == "full" else 0.0)
        self.I, self.b_i = self._solve_side(self._by_item, y_items, self.U[users])
        y_users = r - self.mu - (self.b_i[items] if self.bias == "full" else 0.0)
        self.U, self.b_u = self._solve_side(self._by_user, y_users, self.I[items])

    def _solve_side(self, groups, y, X):
        """Pour chaque ligne j : w_j = argmin sum_{n ∈ groups[j]} (y_n - <w_j, x_n>)^2 + pénalité.

        X[n] est le vecteur de facteurs de « l'autre côté » de la note n (fixé)."""
        k = self.rank
        reg = np.full(k, float(self.lam))
        if self.bias == "full":
            X = np.hstack([X, np.ones((len(X), 1))])
            reg = np.append(reg, self.lam_bias)
        reg = np.diag(reg) + 1e-9 * np.eye(len(reg))  # évite une matrice singulière si lam = 0
        W = np.zeros((len(groups), len(reg)))
        for j, idx in enumerate(groups):
            if len(idx):
                Xj = X[idx]
                W[j] = np.linalg.solve(Xj.T @ Xj + reg, Xj.T @ y[idx])
        b = W[:, k] if self.bias == "full" else np.zeros(len(groups))
        return W[:, :k], b


def _observed(R):
    """Indices (users, items) et valeurs des notes observées de R."""
    users, items = np.nonzero(~np.isnan(R))
    return users, items, R[users, items]


def _groups(index, size):
    """groups[j] = positions n des notes telles que index[n] == j."""
    order = np.argsort(index, kind="stable")
    return np.split(order, np.cumsum(np.bincount(index, minlength=size))[:-1])


def _scatter_sum(index, values, size):
    """out[j] = somme des values[n] tels que index[n] == j  (values 1D ou 2D)."""
    if values.ndim == 1:
        return np.bincount(index, weights=values, minlength=size)
    return np.stack([np.bincount(index, weights=values[:, c], minlength=size)
                     for c in range(values.shape[1])], axis=1)
