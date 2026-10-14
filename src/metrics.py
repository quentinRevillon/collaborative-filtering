"""Métriques d'évaluation, calculées uniquement sur les entrées observées de R_true."""
import numpy as np


def rmse(R_true, R_pred):
    mask = ~np.isnan(R_true)
    return float(np.sqrt(np.mean((R_true[mask] - R_pred[mask]) ** 2)))


def accuracy(R_true, R_pred):
    """Part des notes exactement retrouvées après arrondi au 0.5 le plus proche
    (proxy de la métrique « accuracy for exact ratings » de la plateforme)."""
    mask = ~np.isnan(R_true)
    rounded = np.clip(np.round(R_pred[mask] * 2) / 2, 0.5, 5.0)
    return float(np.mean(rounded == R_true[mask]))
