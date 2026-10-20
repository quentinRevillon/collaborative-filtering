"""Bayesian FM : réglages adaptés à la taille des données de la plateforme.

La plateforme entraîne sur train + test (~63 000 notes), deux fois plus que notre CV sur le
train seul (~25 000 notes par pli) : les réglages optimaux (rang, nombre d'itérations) peuvent
changer. On refait donc la CV 5 plis sur train + test réunis (~50 500 notes par pli).
Pas de test à part : la CV sert à la fois de sélection et d'estimation.

Pour chaque rang : courbe moyenne sur les plis, meilleure itération, et RMSE à l'itération
de la soumission actuelle (118 itérations, burn-in 20) pour comparer.
Sortie : results/bfm_platform.csv (les rangs déjà présents sont conservés, les nouveaux ajoutés).

Lancer depuis la racine du repo :  python -m src.experiments.bfm_platform [--folds 5] [--ranks 30 40]
"""
import argparse

import numpy as np

from src.data import kfold_observed, load_train_test
from src.evaluation import RESULTS_DIR, read_csv, write_csv
from src.models.bfm import BayesianFM
from src.tuning import cv_curves

RANKS = [5, 10, 20]
FIXED = dict(features=("iu", "ii", "side"), n_epochs=500, burn_in=20)
SUBMITTED = dict(rank=5, n_epochs=118)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ranks", type=int, nargs="+", default=RANKS)
    args = parser.parse_args()

    R_train, R_test = load_train_test()
    R_all = np.where(np.isnan(R_train), R_test, R_train)
    folds = kfold_observed(R_all, n_folds=args.folds)
    print(f"Train + test : {int((~np.isnan(R_all)).sum())} notes, CV {args.folds} plis "
          f"(~{int((~np.isnan(folds[0][0])).sum())} notes d'apprentissage par pli)", flush=True)

    rows = []
    at = SUBMITTED["n_epochs"] - 1
    for rank in args.ranks:
        c = cv_curves(BayesianFM, dict(FIXED, rank=rank), folds)
        mean = c.mean(axis=0)
        best = int(np.argmin(mean))
        rows.append(dict(rank=rank, best_epoch=best + 1, cv_rmse=round(float(mean[best]), 4),
                         cv_std=round(float(c[:, best].std()), 4),
                         cv_rmse_at_118=round(float(mean[at]), 4), cv_rmse_at_500=round(float(mean[-1]), 4),
                         per_fold_best=" ".join(f"{v:.4f}" for v in c[:, best]),
                         per_fold_at_118=" ".join(f"{v:.4f}" for v in c[:, at])))
        print(f"  rang {rank:>2} : meilleure it. {best + 1:>3}  cv={mean[best]:.4f} +/- {c[:, best].std():.4f}"
              f"  | it. 118 : {mean[at]:.4f}  | it. 500 : {mean[-1]:.4f}", flush=True)

    if (RESULTS_DIR / "bfm_platform.csv").exists():   # on complète les rangs déjà calculés
        known = {r["rank"]: r for r in read_csv("bfm_platform.csv")}
        known.update({r["rank"]: r for r in rows})
        rows = sorted(known.values(), key=lambda r: r["rank"])
    write_csv("bfm_platform.csv", rows)

    best_row = min(rows, key=lambda r: r["cv_rmse"])
    print(f"\nMeilleure config (tous rangs) : rang {best_row['rank']}, {best_row['best_epoch']} it. "
          f"-> cv={best_row['cv_rmse']} +/- {best_row['cv_std']}  (par pli : {best_row['per_fold_best']})")
    sub = next((r for r in rows if r["rank"] == SUBMITTED["rank"]), None)
    if sub is not None:
        print(f"Soumission actuelle (rang 5, 118 it.) : cv={sub['cv_rmse_at_118']}")


if __name__ == "__main__":
    main()
