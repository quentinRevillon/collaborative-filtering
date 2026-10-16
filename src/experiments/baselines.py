"""Baselines moyenne / biais sur le split local train -> test.

Protocole :
  1. les lambdas sont choisis par validation croisée 5 plis sur le train ;
  2. chaque modèle est réentraîné sur tout le train et évalué une fois sur le test.
Sorties : results/baselines_grid.csv (CV), results/baselines_test.csv (test).

Lancer depuis la racine du repo :  python -m src.experiments.baselines [--folds 5]
"""
import argparse

from src.data import kfold_observed, load_train_test
from src.evaluation import evaluate_on_test, write_csv
from src.models import Biases, GlobalMean
from src.tuning import best_params, grid_search

LAMBDAS = [0, 1, 2, 3, 5, 10, 20]

# variante -> (grille CV, paramètres fixés)
VARIANTS = {
    "item": ({"lam_i": LAMBDAS}, {"use_user": False}),
    "user": ({"lam_u": LAMBDAS}, {"use_item": False}),
    "both": ({"lam_u": LAMBDAS, "lam_i": LAMBDAS}, {}),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    R_train, R_test = load_train_test()
    folds = kfold_observed(R_train, n_folds=args.folds)

    print(f"Reglage des lambdas par CV {args.folds} plis :")
    all_rows, best = [], {}
    for variant, (grid, fixed) in VARIANTS.items():
        print(f"  Biais {variant}")
        rows = grid_search(Biases, grid, folds, fixed=fixed, verbose=variant != "both")
        all_rows += [{"variant": variant, **r} for r in rows]
        best[variant] = rows[0]
        print(f"    -> meilleur : {best_params(rows[0])}  cv={rows[0]['cv_rmse']} +/- {rows[0]['cv_std']}")
    write_csv("baselines_grid.csv", all_rows)

    b = {v: best_params(r) for v, r in best.items()}
    evaluate_on_test({
        "Moyenne globale": GlobalMean(),
        "Biais film (mu + b_i)": Biases(use_user=False),
        "Biais user (mu + b_u)": Biases(use_item=False),
        "Biais user+film, sequentiel": Biases(n_iter=1),
        "Biais user+film, moindres carres": Biases(n_iter=20),
        f"Biais film regularise (lam_i={b['item']['lam_i']})": Biases(**b["item"]),
        f"Biais user regularise (lam_u={b['user']['lam_u']})": Biases(**b["user"]),
        f"Biais regularises (lam_u={b['both']['lam_u']}, lam_i={b['both']['lam_i']})": Biases(**b["both"]),
    }, R_train, R_test, csv_name="baselines_test.csv")


if __name__ == "__main__":
    main()
