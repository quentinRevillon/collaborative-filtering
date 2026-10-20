"""Bayesian Factorization Machine (myfm) : apport de chaque groupe de features, puis rang.

Protocole (même format que les autres expériences) :
  1. CV 5 plis sur le train, rang 10 : features {aucune (= BPMF avec biais), genres,
     implicite iu+ii, tout} ; le nombre d'itérations est choisi sur la courbe moyenne
     (RMSE de la moyenne des échantillons après burn-in) ;
  2. CV sur le rang avec les meilleures features ;
  3. courbe sur le pli 1 : échantillon seul, moyenne des échantillons, alpha tiré ;
  4. la meilleure config est réentraînée sur tout le train et évaluée sur le test, avec la
     baseline biais et BPMF en référence.
Genres : data/namesngenre2.npy (fichier corrigé).
Sorties : results/bfm_grid.csv, results/bfm_curve.csv, results/bfm_test.csv.

Lancer depuis la racine du repo :  python -m src.experiments.bfm [--folds 5] [--skip-cv]
(--skip-cv : reprend la meilleure config de results/bfm_grid.csv et ne fait que les étapes 3-4)
"""
import argparse

from src.data import kfold_observed, load_train_test
from src.evaluation import evaluate_on_test, read_csv, write_csv
from src.models import Biases
from src.models.bfm import BayesianFM
from src.models.bpmf import BPMF
from src.tuning import best_params, grid_search

FEATURE_SETS = {"aucune (= BPMF)": (), "genres": ("side",), "implicite": ("iu", "ii"),
                "tout": ("iu", "ii", "side")}
RANKS = [5, 10, 20]
FIXED = dict(n_epochs=500, burn_in=20)
CHECKPOINTS = [1, 20, 50, 100, 200, 300, 500]


def run_cv(folds):
    """Étapes 1-2 ; renvoie (nom des features retenues, hyperparamètres de la meilleure config)."""
    rows = []
    print(f"1. Features, rang 10, CV {len(folds)} plis :")
    for name, features in FEATURE_SETS.items():
        row = grid_search(BayesianFM, {"rank": [10]}, folds, fixed=dict(FIXED, features=features),
                          verbose=False)[0]
        rows.append({"step": "features", "variant": name, **row})
        print(f"    {name:<16} cv={row['cv_rmse']:.4f} +/- {row['cv_std']:.4f}  ({row['n_epochs']} it.)",
              flush=True)
    best_name = min((r for r in rows if r["step"] == "features"), key=lambda r: r["cv_rmse"])["variant"]
    features = FEATURE_SETS[best_name]

    print(f"2. Rang, features '{best_name}' :")
    for row in grid_search(BayesianFM, {"rank": [k for k in RANKS if k != 10]}, folds,
                           fixed=dict(FIXED, features=features)):
        rows.append({"step": "rank", "variant": best_name, **row})
    best = min((r for r in rows if r["variant"] == best_name), key=lambda r: r["cv_rmse"])
    write_csv("bfm_grid.csv", [{**r, "features": "+".join(r["features"]) or "aucune"} for r in rows])
    p = _params(best)
    p["features"] = features
    print(f"  -> meilleur : {best_name}, rang {p['rank']} ({p['n_epochs']} it.)  "
          f"cv={best['cv_rmse']} +/- {best['cv_std']}")
    return best_name, p


def best_from_csv():
    """Meilleure config de results/bfm_grid.csv (pour ne refaire que les étapes 3-4)."""
    best = min(read_csv("bfm_grid.csv"), key=lambda r: r["cv_rmse"])
    p = _params(best)
    p["features"] = () if p["features"] == "aucune" else tuple(p["features"].split("+"))
    print(f"Meilleure config (results/bfm_grid.csv) : {best['variant']}, {p}  cv={best['cv_rmse']}")
    return best["variant"], p


def _params(row):
    """Hyperparamètres d'une ligne de grille (sans le score ni l'étiquette d'étape)."""
    return {k: v for k, v in best_params(row).items() if k != "step"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--skip-cv", action="store_true")
    args = parser.parse_args()

    R_train, R_test = load_train_test()
    folds = kfold_observed(R_train, n_folds=args.folds)
    best_name, p = best_from_csv() if args.skip_cv else run_cv(folds)

    print("\n3. Courbe sur le pli 1 (echantillon seul vs moyenne, alpha tire) :")
    h = BayesianFM(**dict(p, n_epochs=FIXED["n_epochs"])).fit(*folds[0]).history
    write_csv("bfm_curve.csv", [dict(epoch=e, val_rmse_sample=round(s, 4), val_rmse_mean=round(m, 4),
                                     alpha=round(a, 3))
                                for e, s, m, a in zip(h["epoch"], h["val_rmse_sample"], h["val_rmse"],
                                                      h["alpha"])])
    for e in CHECKPOINTS:
        print(f"  it. {e:>3} : echantillon {h['val_rmse_sample'][e - 1]:.4f}   "
              f"moyenne {h['val_rmse'][e - 1]:.4f}   alpha {h['alpha'][e - 1]:.2f}", flush=True)

    evaluate_on_test({
        "Baseline biais regularises": Biases(lam_u=5, lam_i=3),
        "BPMF avec biais (rang 10, alpha 1.5, 133 it.)": BPMF(bias="full", rank=10, alpha=1.5, n_epochs=133),
        f"Bayesian FM {best_name} (rang {p['rank']}, {p['n_epochs']} it.)": BayesianFM(**p),
    }, R_train, R_test, csv_name="bfm_test.csv")


if __name__ == "__main__":
    main()
