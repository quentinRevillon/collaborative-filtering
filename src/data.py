"""Chargement des matrices de notes.

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


def split_observed(R, frac=0.1, seed=123):
    """Sépare aléatoirement les entrées observées de R en deux matrices disjointes
    (R_fit, R_val), par ex. pour régler les hyperparamètres sans toucher au test."""
    rng = np.random.default_rng(seed)
    users, items = np.nonzero(~np.isnan(R))
    pick = rng.random(len(users)) < frac
    R_fit, R_val = R.copy(), np.full_like(R, np.nan)
    R_val[users[pick], items[pick]] = R[users[pick], items[pick]]
    R_fit[users[pick], items[pick]] = np.nan
    return R_fit, R_val
