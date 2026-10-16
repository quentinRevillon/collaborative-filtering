"""Test simple de BPMF (Salakhutdinov & Mnih 2008) : variante du papier vs variante avec biais.

Pas de lambda à régler (la régularisation est tirée par Gibbs) ; hyperpriors du papier
(beta0 = 2, nu0 = D, W0 = Id), alpha = 1.5 (réglé par CV) ; on ne fait varier que la variante et le rang.

Protocole :
  1. CV 5 plis sur le train pour 3 configs ; le nombre d'itérations est choisi sur la
     courbe moyenne (RMSE de la moyenne des échantillons après burn-in) ;
  2. courbe d'un pli : RMSE d'un échantillon seul vs RMSE de la moyenne des échantillons ;
  3. la meilleure config est réentraînée sur tout le train et évaluée sur le test, avec la
     baseline biais et le meilleur kNN films en référence.
Sorties : results/bpmf_grid.csv, results/bpmf_curve.csv, results/bpmf_test.csv.

Lancer depuis la racine du repo :  python -m src.experiments.bpmf [--folds 5]
"""
import argparse

from src.data import kfold_observed, load_train_test
from src.evaluation import evaluate_on_test, write_csv
from src.models import Biases, KNNResidual
from src.models.bpmf import BPMF
from src.tuning import best_params, grid_search

FIXED = dict(n_epochs=150, burn_in=20, alpha=1.5)  # alpha réglé par src/experiments/bpmf_alpha.py
CONFIGS = [("mean", [10]), ("full", [10, 20])]   # (variante, rangs testés)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    R_train, R_test = load_train_test()
    folds = kfold_observed(R_train, n_folds=args.folds)

    rows = []
    for bias, ranks in CONFIGS:
        print(f"BPMF '{bias}', CV {args.folds} plis :")
        rows += [{"variant": bias, **r} for r in
                 grid_search(BPMF, {"rank": ranks}, folds, fixed=dict(FIXED, bias=bias))]
    rows.sort(key=lambda r: r["cv_rmse"])
    write_csv("bpmf_grid.csv", rows)
    p = best_params(rows[0])
    print(f"  -> meilleur : {p['bias']} rang {p['rank']} ({p['n_epochs']} it.)  "
          f"cv={rows[0]['cv_rmse']} +/- {rows[0]['cv_std']}")

    print("\nCourbe sur le pli 1 (echantillon seul vs moyenne des echantillons) :")
    h = BPMF(**dict(p, n_epochs=FIXED["n_epochs"])).fit(*folds[0]).history
    write_csv("bpmf_curve.csv", [dict(epoch=e, val_rmse_sample=round(s, 4), val_rmse_mean=round(m, 4))
                                 for e, s, m in zip(h["epoch"], h["val_rmse_sample"], h["val_rmse"])])
    for e in [1, 10, 20, 21, 30, 50, 100, 150]:
        print(f"  it. {e:>3} : echantillon {h['val_rmse_sample'][e - 1]:.4f}   moyenne {h['val_rmse'][e - 1]:.4f}")

    evaluate_on_test({
        "Baseline biais regularises": Biases(lam_u=5, lam_i=3),
        "kNN films (k=20, shrink=100, damp=0.5)": KNNResidual(kind="item", k=20, shrink=100, damp=0.5),
        f"BPMF {p['bias']} (rang {p['rank']}, {p['n_epochs']} it., burn-in {p['burn_in']})": BPMF(**p),
    }, R_train, R_test, csv_name="bpmf_test.csv")


if __name__ == "__main__":
    main()
