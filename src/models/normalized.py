"""Normalisation des notes par user autour d'un modèle quelconque.

Pour chaque user, moyenne m_u et écart-type s_u de ses notes d'entraînement, rétrécis vers les
valeurs globales (mu, sigma) avec un poids de `shrink` pseudo-notes :
    m_u = (sum_i r_ui + shrink * mu) / (n_u + shrink)
    s_u = sqrt((sum_i (r_ui - m_u)^2 + shrink * sigma^2) / (n_u + shrink))
Le modèle de base apprend sur z_ui = (r_ui - m_u) / s_u, puis r̂_ui = m_u + s_u * ẑ_ui.
Avec center=False, m_u = mu pour tous (réduction seule, la moyenne de l'user reste au modèle).

Intérêt : un modèle à bruit homogène (un seul alpha pour la Bayesian FM) donne alors moins de
poids aux users dont les notes sont très dispersées, qui concentrent l'erreur.
"""
import numpy as np

from src.models.base import Model, clip_ratings


class UserNormalized(Model):
    def __init__(self, base, shrink=5.0, center=True, scale=True, clip=True):
        self.base, self.shrink, self.center, self.scale, self.clip = base, shrink, center, scale, clip

    def fit(self, R, R_val=None):
        mask = ~np.isnan(R)
        users = np.nonzero(mask)[0]
        r = R[mask]
        mu, sigma = r.mean(), r.std()
        n_u = mask.sum(1)
        R0 = np.where(mask, R, 0.0)
        m_u = (R0.sum(1) + self.shrink * mu) / (n_u + self.shrink) if self.center else np.full(len(n_u), mu)
        if self.scale:
            dev = np.where(mask, (R0 - m_u[:, None]) ** 2, 0.0).sum(1)
            s_u = np.sqrt((dev + self.shrink * sigma ** 2) / (n_u + self.shrink))
        else:
            s_u = np.ones(len(n_u))
        self._m, self._s = m_u, s_u
        Z = np.where(mask, (R - m_u[:, None]) / s_u[:, None], np.nan)
        self.base.fit(Z)
        return self

    def predict(self):
        R_hat = self._m[:, None] + self._s[:, None] * self.base.predict()
        return clip_ratings(R_hat) if self.clip else R_hat
