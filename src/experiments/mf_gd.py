"""Factorisation de matrice par descente de gradient, sur le split local train -> test.

Trois variantes, du modèle du cours vers le modèle complet :
    none : R ≈ U I^T            mean : R ≈ mu + U I^T            full : R ≈ mu + b_u + b_i + U I^T

Protocole :
  1. grille (rang k, lambda) par validation croisée 5 plis sur le train ; pour chaque
     config, le nombre d'époques est choisi sur la courbe de validation moyenne ;
  2. la meilleure config de chaque variante est réentraînée sur tout le train, puis
     évaluée sur le test (avec la baseline biais en référence).
Sorties : results/mf_gd_grid.csv (CV), results/mf_gd_test.csv (test).
Long : ~45 configs x 5 plis x 800 époques, compter ~30 min.

Lancer depuis la racine du repo :  python -m src.experiments.mf_gd [--folds 5]
"""
import argparse

from src.data import kfold_observed, load_train_test
from src.evaluation import best_row, evaluate_on_test, write_csv
from src.models import Biases, MatrixFactorizationGD
from src.tuning import best_params, grid_search

GRID = {"rank": [2, 10, 50], "lam": [1, 5, 10, 15, 20]}
GD = dict(n_epochs=800, lr=3e-4, momentum=0.9)
# Le modèle pur doit représenter toute la note (~3.5) avec <u_u, i_i> : init plus large.
INIT_STD = {"none": 0.5, "mean": 0.1, "full": 0.1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    R_train, R_test = load_train_test()
    folds = kfold_observed(R_train, n_folds=args.folds)

    all_rows, models = [], {}
    baseline = best_row("baselines_grid.csv", "both")
    if baseline:
        models["Baseline biais regularises"] = Biases(**best_params(baseline))

    for bias in ["none", "mean", "full"]:
        print(f"Grille MF-GD '{bias}', CV {args.folds} plis :")
        fixed = dict(GD, bias=bias, init_std=INIT_STD[bias])
        rows = grid_search(MatrixFactorizationGD, GRID, folds, fixed=fixed)
        all_rows += [{"variant": bias, **r} for r in rows]
        p = best_params(rows[0])
        models[f"MF-GD {bias} (k={p['rank']}, lam={p['lam']}, {p['n_epochs']} ep.)"] = \
            MatrixFactorizationGD(**p)
    write_csv("mf_gd_grid.csv", all_rows)

    evaluate_on_test(models, R_train, R_test, csv_name="mf_gd_test.csv")


if __name__ == "__main__":
    main()
