"""kNN (Pearson sur les résidus, avec shrink) ajouté au modèle de biais, films vs users.

Protocole :
  1. CV 5 plis sur le train : pour chaque pli et chaque type (films / users), les
     corrélations sont calculées une fois ; k, shrink et damp ne jouent qu'à la
     prédiction, donc toute la grille est balayée rapidement (KNNResidual.val_rmse_grid) ;
  2. la meilleure config de chaque type est réentraînée sur tout le train et évaluée
     sur le test, avec la baseline biais en référence.
Sorties : results/knn_grid.csv (CV), results/knn_test.csv (test).

Lancer depuis la racine du repo :  python -m src.experiments.knn [--folds 5]
"""
import argparse

import numpy as np

from src.data import kfold_observed, load_train_test
from src.evaluation import evaluate_on_test, write_csv
from src.metrics import rmse
from src.models import Biases
from src.models.knn import KNNResidual

KS = [5, 10, 20, 40, 80]
SHRINKS = [0, 10, 25, 50, 100, 200]
DAMPS = [0, 0.5, 1, 2, 5, 10]
BASELINE = dict(lam_u=5, lam_i=3)  # meilleurs lambdas de src/experiments/baselines.py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    R_train, R_test = load_train_test()
    folds = kfold_observed(R_train, n_folds=args.folds)

    base_scores = [rmse(R_val, Biases(**BASELINE).fit(R_fit).predict()) for R_fit, R_val in folds]
    print(f"Baseline biais : cv={np.mean(base_scores):.4f} +/- {np.std(base_scores):.4f}")

    rows, best = [], {}
    for kind in ["item", "user"]:
        scores = {}  # (shrink, k, damp) -> [rmse par pli]
        for R_fit, R_val in folds:
            model = KNNResidual(kind=kind, **BASELINE).fit(R_fit)
            for shrink in SHRINKS:
                model.shrink = shrink
                for (k, damp), v in model.val_rmse_grid(R_val, KS, DAMPS).items():
                    scores.setdefault((shrink, k, damp), []).append(v)
        kind_rows = [dict(kind=kind, shrink=s, k=k, damp=d, cv_rmse=round(float(np.mean(v)), 4),
                          cv_std=round(float(np.std(v)), 4))
                     for (s, k, d), v in scores.items()]
        rows += kind_rows
        best[kind] = min(kind_rows, key=lambda r: r["cv_rmse"])

        print(f"\nkNN {kind} - meilleure RMSE CV par (shrink, damp), sur k :")
        print("  shrink \\ damp " + "".join(f"{d:>8}" for d in DAMPS))
        for s in SHRINKS:
            cells = [min(r["cv_rmse"] for r in kind_rows if r["shrink"] == s and r["damp"] == d)
                     for d in DAMPS]
            print(f"  {s:>13} " + "".join(f"{c:>8.4f}" for c in cells))
        b = best[kind]
        print(f"  -> meilleur : shrink={b['shrink']} k={b['k']} damp={b['damp']}  "
              f"cv={b['cv_rmse']} +/- {b['cv_std']}")
    write_csv("knn_grid.csv", rows)

    models = {"Baseline biais regularises": Biases(**BASELINE)}
    for kind, b in best.items():
        name = f"Biais + kNN {kind} (k={b['k']}, shrink={b['shrink']}, damp={b['damp']})"
        models[name] = KNNResidual(kind=kind, k=b["k"], shrink=b["shrink"], damp=b["damp"], **BASELINE)
    evaluate_on_test(models, R_train, R_test, csv_name="knn_test.csv")


if __name__ == "__main__":
    main()
