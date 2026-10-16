"""Chargement des matrices de notes et découpages pour la validation.

Convention du dataset : matrice (n_users, n_items) = (610, 4980),
NaN = note non observée, notes dans {0.5, 1.0, ..., 5.0}.
"""
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_ratings(name):
    """Charge `data/<name>.npy` (ou un chemin explicite vers un .npy)."""
    path = Path(name)
    if not path.suffix:
        path = DATA_DIR / f"{name}.npy"
    return np.load(path)


def load_train_test():
    return load_ratings("ratings_train"), load_ratings("ratings_test")


def split_observed(R, frac=0.1, seed=0):
    """Sépare aléatoirement les entrées observées de R en deux matrices disjointes
    (R_fit, R_val), `frac` des notes allant dans R_val."""
    users, items = np.nonzero(~np.isnan(R))
    pick = np.random.default_rng(seed).random(len(users)) < frac
    return _split(R, users[pick], items[pick])


def kfold_observed(R, n_folds=5, seed=0):
    """Validation croisée sur les notes observées : renvoie n_folds couples (R_fit, R_val).

    Chaque note observée est en validation dans exactement un pli ; avec n_folds=5,
    chaque pli apprend sur 80 % des notes et valide sur les 20 % restants.
    """
    users, items = np.nonzero(~np.isnan(R))
    fold = np.random.default_rng(seed).permutation(len(users)) % n_folds
    return [_split(R, users[fold == f], items[fold == f]) for f in range(n_folds)]


def _split(R, val_users, val_items):
    """Déplace les notes (val_users, val_items) de R vers une matrice de validation."""
    R_fit, R_val = R.copy(), np.full_like(R, np.nan)
    R_val[val_users, val_items] = R[val_users, val_items]
    R_fit[val_users, val_items] = np.nan
    return R_fit, R_val
