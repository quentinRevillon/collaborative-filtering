"""Effet de l'initialisation sur BPMF : départ aléatoire vs quelques itérations d'ALS.

Même config que la meilleure de src/experiments/bpmf.py (lue dans results/bpmf_grid.csv),
on ne fait varier que `init_epochs` (nombre d'itérations d'ALS avant Gibbs, 0 = aléatoire).
Pour chaque init, CV 5 plis : courbes moyennes sur les plis, à chaque itération de Gibbs,
de la RMSE de validation d'un échantillon seul et de la moyenne des échantillons.
Sorties : results/bpmf_init_curves.csv, results/bpmf_init.log (via la console).

Lancer depuis la racine du repo :  python -m src.experiments.bpmf_init [--folds 5]
"""
import argparse

import numpy as np

from src.data import kfold_observed, load_train_test
from src.evaluation import best_row, write_csv
from src.models.bpmf import BPMF
from src.tuning import best_params

INITS = [0, 2, 10, 30]
CHECKPOINTS = [1, 10, 20, 30, 50, 100, 150]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    R_train, _ = load_train_test()
    folds = kfold_observed(R_train, n_folds=args.folds)
    row = best_row("bpmf_grid.csv")
    base = best_params(row) if row else dict(bias="full", rank=10, burn_in=20, alpha=2.0)
    base = {k: v for k, v in base.items() if k != "n_epochs"} | dict(n_epochs=150)
    print(f"Config : {base}")

    rows = []
    for init in INITS:
        hist = [BPMF(**base, init_epochs=init).fit(R_fit, R_val).history for R_fit, R_val in folds]
        mean_curve = np.mean([h["val_rmse"] for h in hist], axis=0)
        sample_curve = np.mean([h["val_rmse_sample"] for h in hist], axis=0)
        rows += [dict(init_epochs=init, epoch=e + 1, val_rmse_mean=round(float(m), 4),
                      val_rmse_sample=round(float(s), 4))
                 for e, (m, s) in enumerate(zip(mean_curve, sample_curve))]
        label = "aleatoire" if init == 0 else f"ALS x{init}"
        print(f"\ninit {label:<10} (moyenne sur {args.folds} plis)")
        print("  iteration Gibbs : " + "".join(f"{c:>8}" for c in CHECKPOINTS))
        print("  echantillon     : " + "".join(f"{sample_curve[c - 1]:>8.4f}" for c in CHECKPOINTS))
        print("  moyenne         : " + "".join(f"{mean_curve[c - 1]:>8.4f}" for c in CHECKPOINTS))
    write_csv("bpmf_init_curves.csv", rows)


if __name__ == "__main__":
    main()
