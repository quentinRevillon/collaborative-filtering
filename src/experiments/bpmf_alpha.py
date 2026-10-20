"""BPMF : réglage de alpha (précision du bruit sur les notes), le seul hyperparamètre qui
joue le rôle d'une régularisation (les hyperpriors beta0, nu0, W0, mu0 sont peu informatifs).

Quatre modes ; les trois premiers en CV 5 plis sur le train, sans toucher au test :
  - grille : alpha fixé, grille 1D, même graine pour tous les alpha (mêmes tirages
    aléatoires d'une config à l'autre : les écarts reflètent mieux l'effet de alpha que
    le bruit Monte-Carlo). Les résultats sont ajoutés à results/bpmf_alpha_grid.csv ;
  - --gibbs : alpha tiré à chaque itération (prior Gamma(a0, b0)) ; on note la RMSE CV
    et la valeur de alpha au fil des itérations -> results/bpmf_alpha_gibbs.csv.
  - --long : alpha = 1.5 vs 2 dans la config du generate.py de main (rang 20, burn-in 100,
    400 itérations), en CV -> results/bpmf_alpha_long.csv ;
  - --test : évaluation finale sur le test (alpha = 2 du papier vs alpha = 1.5 réglé par CV),
    chaque config avec le nombre d'itérations choisi par la CV -> results/bpmf_alpha_test.csv.
Config commune : variante avec biais, rang 10, 150 itérations, burn-in 20.

Lancer depuis la racine du repo :
    python -m src.experiments.bpmf_alpha [--alphas 1.25 1.4 1.6 1.75]
    python -m src.experiments.bpmf_alpha --gibbs
    python -m src.experiments.bpmf_alpha --long
    python -m src.experiments.bpmf_alpha --test
"""
import argparse

import numpy as np

from src.data import kfold_observed, load_train_test
from src.evaluation import RESULTS_DIR, evaluate_on_test, read_csv, write_csv
from src.models import Biases, KNNResidual
from src.models.bpmf import BPMF
from src.tuning import cv_curves, grid_search

DEFAULT_ALPHAS = [0.5, 1, 1.5, 2, 3, 5, 8]
FIXED = dict(bias="full", rank=10, n_epochs=150, burn_in=20, seed=0)
CHECKPOINTS = [1, 5, 10, 20, 50, 100, 150]


def run_grid(folds, alphas):
    print(f"BPMF, grille sur alpha, CV {len(folds)} plis ({FIXED}) :")
    rows = grid_search(BPMF, {"alpha": alphas}, folds, fixed=FIXED)
    path = "bpmf_alpha_grid.csv"
    if (RESULTS_DIR / path).exists():   # on complète la grille existante
        known = {r["alpha"]: r for r in read_csv(path)}
        known.update({r["alpha"]: r for r in rows})
        rows = list(known.values())
    rows.sort(key=lambda r: r["alpha"])
    write_csv(path, rows)
    best = min(rows, key=lambda r: r["cv_rmse"])
    print(f"  -> meilleur sur toute la grille : alpha={best['alpha']} ({best['n_epochs']} it.)  "
          f"cv={best['cv_rmse']} +/- {best['cv_std']}")


def run_gibbs(folds):
    print(f"BPMF, alpha tire dans Gibbs (prior Gamma(1, 1), depart alpha=2), CV {len(folds)} plis :")
    hist = [BPMF(**FIXED, alpha="gibbs").fit(R_fit, R_val).history for R_fit, R_val in folds]
    val = np.mean([h["val_rmse"] for h in hist], axis=0)
    alpha = np.array([h["alpha"] for h in hist])            # (plis, itérations)
    train = np.mean([h["train_rmse"] for h in hist], axis=0)
    best = int(np.argmin(val))
    per_fold = [h["val_rmse"][best] for h in hist]
    print(f"  cv={val[best]:.4f} +/- {np.std(per_fold):.4f}  ({best + 1} it.)")
    print("  iteration      : " + "".join(f"{c:>8}" for c in CHECKPOINTS))
    print("  alpha (moyen)  : " + "".join(f"{alpha[:, c - 1].mean():>8.3f}" for c in CHECKPOINTS))
    print("  1/sqrt(alpha)  : " + "".join(f"{1 / np.sqrt(alpha[:, c - 1].mean()):>8.3f}" for c in CHECKPOINTS))
    print("  RMSE train     : " + "".join(f"{train[c - 1]:>8.3f}" for c in CHECKPOINTS))
    print("  RMSE val (moy.): " + "".join(f"{val[c - 1]:>8.4f}" for c in CHECKPOINTS))
    after = alpha[:, FIXED["burn_in"]:]
    print(f"  alpha apres burn-in : moyenne {after.mean():.3f}, ecart-type {after.std():.3f}")
    write_csv("bpmf_alpha_gibbs.csv", [
        dict(epoch=e + 1, alpha_mean=round(float(alpha[:, e].mean()), 4),
             train_rmse=round(float(train[e]), 4), val_rmse=round(float(val[e]), 4))
        for e in range(len(val))])


def run_long(folds, alphas=(1.5, 2)):
    """Configuration du generate.py de main (rang 20, burn-in 100, 400 itérations) : RMSE CV
    à la dernière itération (ce que fait generate.py) et à la meilleure itération."""
    config = dict(FIXED, rank=20, burn_in=100, n_epochs=400)
    print(f"BPMF, config longue {config}, CV {len(folds)} plis :")
    rows = []
    for alpha in alphas:
        curves = cv_curves(BPMF, dict(config, alpha=alpha), folds)
        mean = curves.mean(axis=0)
        best = int(np.argmin(mean))
        row = dict(alpha=alpha, rank=20, burn_in=100,
                   cv_rmse_last=round(float(mean[-1]), 4), cv_std_last=round(float(curves[:, -1].std()), 4),
                   best_epoch=best + 1, cv_rmse_best=round(float(mean[best]), 4),
                   per_fold_last=" ".join(f"{v:.4f}" for v in curves[:, -1]))
        rows.append(row)
        print(f"  alpha={alpha}: iteration 400 cv={row['cv_rmse_last']} +/- {row['cv_std_last']}"
              f" | meilleure iteration {row['best_epoch']} cv={row['cv_rmse_best']}"
              f" | par pli (it. 400) {row['per_fold_last']}", flush=True)
    write_csv("bpmf_alpha_long.csv", rows)


def run_test(R_train, R_test):
    grid = {r["alpha"]: r for r in read_csv("bpmf_alpha_grid.csv")}
    models = {
        "Baseline biais regularises": Biases(lam_u=5, lam_i=3),
        "kNN films (k=20, shrink=100, damp=0.5)": KNNResidual(kind="item", k=20, shrink=100, damp=0.5),
    }
    for alpha, label in [(2, "papier"), (1.5, "regle par CV")]:
        n_epochs = grid[alpha]["n_epochs"]
        models[f"BPMF full, alpha={alpha} ({label}, {n_epochs} it.)"] = BPMF(
            **dict(FIXED, alpha=alpha, n_epochs=n_epochs))
    evaluate_on_test(models, R_train, R_test, csv_name="bpmf_alpha_test.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    parser.add_argument("--gibbs", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--long", action="store_true")
    args = parser.parse_args()

    R_train, R_test = load_train_test()
    if args.test:
        run_test(R_train, R_test)
        return
    folds = kfold_observed(R_train, n_folds=args.folds)
    if args.long:
        run_long(folds)
    elif args.gibbs:
        run_gibbs(folds)
    else:
        run_grid(folds, args.alphas)


if __name__ == "__main__":
    main()
