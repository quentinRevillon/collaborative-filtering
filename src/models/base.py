"""Interface commune à tous les modèles de complétion de la matrice de notes."""
import numpy as np

RATING_MIN, RATING_MAX = 0.5, 5.0


class Model:
    """Un modèle se construit avec ses hyperparamètres, puis :
        fit(R, R_val=None) -> self    R : (n_users, n_films), NaN = note non observée
        predict()          -> matrice complète (n_users, n_films)

    Modèle itératif (iterative = True) : il a un hyperparamètre `n_epochs`, et
    fit(R, R_val) remplit self.history["val_rmse"] (une valeur par époque). src.tuning
    s'en sert pour choisir le nombre d'époques (early stopping).

    param_grid() : grille de recherche par défaut, utilisée par src.tuning si
    l'expérience n'en fournit pas d'autre.
    """

    iterative = False

    def fit(self, R, R_val=None):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    @staticmethod
    def param_grid():
        return {}


def clip_ratings(R_hat):
    return np.clip(R_hat, RATING_MIN, RATING_MAX)
