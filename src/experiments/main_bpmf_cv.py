"""Vérification du BPMF de main (src/bpmf.py) sur nos plis : quatre configurations.

Configs : celle de main (rang 20, alpha 2, burn-in 100, 300 échantillons), la même avec
alpha 1.5, et rang 10 / burn-in 20 / 120 échantillons avec alpha 2 puis 1.5.
CV 5 plis sur ratings_train, mêmes plis que nos autres expériences (kfold_observed, graine 0).
Nécessite torch et scipy (dépendances du BPMF de main).
Sorties : results/main_bpmf_cv.csv (+ log console).

Lancer depuis la racine du repo :  python -m src.experiments.main_bpmf_cv [--main ../a1-main]
"""
import time

import numpy as np

from src.data import kfold_observed, load_train_test
from src.evaluation import write_csv
from src.experiments.main_code import load_main_module, main_path_from_args

CONFIGS = {
    "main actuel (r20, a2, burn 100, 300)": dict(rank=20, n_samples=300, burn_in=100, alpha=2.0),
    "r20, a1.5, burn 100, 300": dict(rank=20, n_samples=300, burn_in=100, alpha=1.5),
    "r10, a2, burn 20, 120": dict(rank=10, n_samples=120, burn_in=20, alpha=2.0),
    "propose (r10, a1.5, burn 20, 120)": dict(rank=10, n_samples=120, burn_in=20, alpha=1.5),
}
REFERENCE = "main actuel (r20, a2, burn 100, 300)"


def main():
    args = main_path_from_args(__doc__.splitlines()[0])
    bpmf = load_main_module(args.main, "bpmf")
    data = load_main_module(args.main, "data")

    R_train, _ = load_train_test()
    folds = kfold_observed(R_train, n_folds=args.folds)
    scores = {name: [] for name in CONFIGS}
    times = {name: [] for name in CONFIGS}
    for f, (R_fit, R_val) in enumerate(folds, 1):
        mask = ~np.isnan(R_val)
        for name, cfg in CONFIGS.items():
            t0 = time.perf_counter()
            M = data.complete(bpmf.fit_bpmf(data.to_coo(R_fit), R_fit.shape, **cfg), R_fit.shape)
            times[name].append(time.perf_counter() - t0)
            scores[name].append(float(np.sqrt(np.mean((R_val[mask] - M[mask]) ** 2))))
        print(f"pli {f} : " + "  ".join(f"{n.split(' (')[0]}={scores[n][-1]:.4f}" for n in CONFIGS),
              flush=True)

    ref = np.array(scores[REFERENCE])
    rows = []
    print(f"\n{'Configuration':<40}{'RMSE CV':>9}{'+/-':>8}{'ecart vs main':>15}{'plis gagnes':>13}{'temps/pli':>11}")
    for name, cfg in CONFIGS.items():
        s = np.array(scores[name])
        diff = s - ref
        rows.append(dict(config=name, **cfg, cv_rmse=round(s.mean(), 4), cv_std=round(s.std(), 4),
                         diff_vs_main=round(diff.mean(), 4), folds_better=int((diff < 0).sum()),
                         sec_per_fold=round(np.mean(times[name]), 1),
                         per_fold=" ".join(f"{v:.4f}" for v in s)))
        print(f"{name:<40}{s.mean():>9.4f}{s.std():>8.4f}{diff.mean():>+15.4f}{int((diff < 0).sum()):>10}/5"
              f"{np.mean(times[name]):>10.1f}s")
    write_csv("main_bpmf_cv.csv", rows)


if __name__ == "__main__":
    main()
