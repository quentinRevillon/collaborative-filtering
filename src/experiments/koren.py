"""Modèle de voisinage de Koren (poids w_ij appris) vs kNN à poids de similarité.

Protocole :
  1. grille (k voisins, lam_w) par validation croisée 5 plis sur le train ; le nombre
     d'itérations de gradient conjugué est choisi sur la courbe de validation moyenne ;
  2. la meilleure config est réentraînée sur tout le train et évaluée sur le test, avec
     la baseline biais et le meilleur kNN sur les résidus (results/knn_grid.csv) en référence.
Sorties : results/koren_grid.csv (CV), results/koren_test.csv (test).

Lancer depuis la racine du repo :  python -m src.experiments.koren [--folds 5]
"""
import argparse

from src.data import kfold_observed, load_train_test
from src.evaluation import RESULTS_DIR, evaluate_on_test, read_csv, write_csv
from src.models import Biases, KNNResidual, KorenNeighborhood
from src.tuning import best_params, grid_search

GRID = {"k": [20, 50, 100, 200], "lam_w": [1, 3, 10, 30, 100]}
FIXED = dict(n_epochs=40, lam_u=5, lam_i=3, shrink=100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    R_train, R_test = load_train_test()
    folds = kfold_observed(R_train, n_folds=args.folds)

    print(f"Grille Koren, CV {args.folds} plis :")
    rows = grid_search(KorenNeighborhood, GRID, folds, fixed=FIXED)
    write_csv("koren_grid.csv", rows)
    p = best_params(rows[0])
    print(f"  -> meilleur : k={p['k']} lam_w={p['lam_w']} ({p['n_epochs']} it.)  "
          f"cv={rows[0]['cv_rmse']} +/- {rows[0]['cv_std']}")

    models = {"Baseline biais regularises": Biases(lam_u=5, lam_i=3)}
    if (RESULTS_DIR / "knn_grid.csv").exists():
        b = min((r for r in read_csv("knn_grid.csv") if r["kind"] == "item"), key=lambda r: r["cv_rmse"])
        models[f"kNN films, poids = similarite (k={b['k']}, shrink={b['shrink']}, damp={b['damp']})"] = \
            KNNResidual(kind="item", k=b["k"], shrink=b["shrink"], damp=b["damp"], lam_u=5, lam_i=3)
    models[f"Koren, poids appris (k={p['k']}, lam_w={p['lam_w']}, {p['n_epochs']} it.)"] = \
        KorenNeighborhood(**p)
    evaluate_on_test(models, R_train, R_test, csv_name="koren_test.csv")


if __name__ == "__main__":
    main()
