"""Factorisation de matrice par ALS, comparée à la descente de gradient (split train -> test).

Même protocole que src/experiments/mf_gd.py :
  1. grille (variante, rang k, lambda) par validation croisée 5 plis sur le train ;
     pour chaque config, le nombre d'itérations est choisi sur la courbe moyenne ;
  2. GD vs ALS sur la meilleure config "full", sur le premier pli : coût C et RMSE
     de validation à chaque époque (pour src/plots/convergence.py) ;
  3. la meilleure config de chaque variante est réentraînée sur tout le train, puis
     évaluée sur le test, avec la baseline biais et la meilleure MF-GD en référence.
Sorties : results/als_grid.csv, results/gd_vs_als_curves.csv, results/als_test.csv.

Lancer depuis la racine du repo :  python -m src.experiments.als [--folds 5]
"""
import argparse

import numpy as np

from src.data import kfold_observed, load_train_test
from src.evaluation import best_row, evaluate_on_test, write_csv
from src.metrics import rmse
from src.models import Biases, MatrixFactorizationALS, MatrixFactorizationGD
from src.tuning import best_params, grid_search

GRID = {"rank": [2, 10, 50], "lam": [1, 5, 10, 15, 20, 30]}
N_ITER = 30
INIT_STD = {"none": 0.5, "mean": 0.1, "full": 0.1}


def convergence_curves(R_fit, R_val, rank, lam, baseline):
    """GD et ALS sur la même config : historique complet pour comparer les vitesses.
    La baseline biais est ajoutée comme une ligne de référence (method="baseline")."""
    runs = {
        "GD": MatrixFactorizationGD(rank=rank, lam=lam, n_epochs=800, lr=3e-4, momentum=0.9),
        "ALS": MatrixFactorizationALS(rank=rank, lam=lam, n_epochs=N_ITER),
    }
    rows = [dict(method="baseline", epoch=0, time=0.0, loss=None,
                 val_rmse=round(rmse(R_val, baseline.fit(R_fit).predict()), 4))]
    for name, model in runs.items():
        h = model.fit(R_fit, R_val).history
        rows += [dict(method=name, epoch=e, time=round(t, 3), loss=round(l, 1), val_rmse=round(v, 4))
                 for e, t, l, v in zip(h["epoch"], h["time"], h["loss"], h["val_rmse"])]
        best = int(np.argmin(h["val_rmse"]))
        print(f"  {name:<4} meilleure val {h['val_rmse'][best]:.4f} a l'epoque {best + 1} "
              f"({h['time'][best]:.1f}s) | cout final {h['loss'][-1]:.0f} ({h['time'][-1]:.1f}s)")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    R_train, R_test = load_train_test()
    folds = kfold_observed(R_train, n_folds=args.folds)

    baseline_row = best_row("baselines_grid.csv", "both")
    baseline = Biases(**best_params(baseline_row)) if baseline_row else Biases(lam_u=5, lam_i=3)
    models = {"Baseline biais regularises": baseline}
    gd_row = best_row("mf_gd_grid.csv", "full")
    if gd_row:
        p = best_params(gd_row)
        models[f"MF-GD full (k={p['rank']}, lam={p['lam']}, {p['n_epochs']} ep.)"] = \
            MatrixFactorizationGD(**p)

    all_rows, best = [], {}
    for bias in ["none", "mean", "full"]:
        print(f"Grille ALS '{bias}', CV {args.folds} plis :")
        fixed = dict(bias=bias, n_epochs=N_ITER, init_std=INIT_STD[bias])
        rows = grid_search(MatrixFactorizationALS, GRID, folds, fixed=fixed)
        all_rows += [{"variant": bias, **r} for r in rows]
        best[bias] = best_params(rows[0])
    write_csv("als_grid.csv", all_rows)

    p = best["full"]
    print(f"\nGD vs ALS a configuration egale (full, k={p['rank']}, lam={p['lam']}), pli 1 :")
    write_csv("gd_vs_als_curves.csv", convergence_curves(*folds[0], p["rank"], p["lam"], baseline))

    for bias, p in best.items():
        models[f"MF-ALS {bias} (k={p['rank']}, lam={p['lam']}, {p['n_epochs']} it.)"] = \
            MatrixFactorizationALS(**p)
    evaluate_on_test(models, R_train, R_test, csv_name="als_test.csv")


if __name__ == "__main__":
    main()
