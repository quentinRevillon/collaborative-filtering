"""Tous les modèles de complétion. Pour ajouter une méthode : un fichier dans ce dossier
(une classe qui hérite de src.models.base.Model), puis une entrée dans MODELS."""
from src.models.baselines import Biases, GlobalMean
from src.models.bfm import BayesianFM
from src.models.bpmf import BPMF
from src.models.knn import KNNResidual
from src.models.koren import KorenNeighborhood
from src.models.mf import MatrixFactorizationALS, MatrixFactorizationGD

MODELS = {
    "global_mean": GlobalMean,
    "biases": Biases,
    "mf_gd": MatrixFactorizationGD,
    "mf_als": MatrixFactorizationALS,
    "knn_residual": KNNResidual,
    "koren": KorenNeighborhood,
    "bpmf": BPMF,
    "bfm": BayesianFM,
}
