"""Factorisation de matrice par ALS, comparée à la descente de gradient (split train -> test).

Même protocole que src/run_mf_gd.py :
  1. grille (variante, rang k, lambda) sur une validation = 10 % des notes de train ;
     pour chaque config on garde la meilleure itération ;
  2. la meilleure config de chaque variante est réentraînée sur tout le train, puis
     évaluée sur le test.
  3. GD vs ALS sur une même config : coût C et RMSE de validation en fonction du temps.
Sorties : results/als_grid.csv, results/gd_vs_als_curves.csv.

Lancer depuis la racine du repo :  python -m src.run_als
"""
import csv
import time
from pathlib import Path

import numpy as np

from src.baselines import Biases
from src.data import load_train_test, split_observed
from src.metrics import accuracy, rmse
from src.mf import MatrixFactorizationALS, MatrixFactorizationGD

RANKS = [2, 10, 50]
LAMBDAS = [1, 5, 10, 15, 20, 30]
N_ITER = 30
# Le modèle pur doit représenter toute la note (~3.5) avec <u_u, i_i> : init plus large.
INIT_STD = {"none": 0.5, "mean": 0.1, "full": 0.1}
# Meilleure config GD trouvée par src/run_mf_gd.py (cf. results/mf_gd_grid.csv)
GD_BEST = dict(rank=50, lam=10, n_epochs=201, lr=3e-4, momentum=0.9)
RESULTS = Path(__file__).resolve().parent.parent / "results"


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def grid_search(R_fit, R_val):
    rows = []
    for bias in ["none", "mean", "full"]:
        for k in RANKS:
            for lam in LAMBDAS:
                model = MatrixFactorizationALS(rank=k, lam=lam, n_epochs=N_ITER, bias=bias,
                                               init_std=INIT_STD[bias])
                h = model.fit(R_fit, R_val).history
                val = np.array(h["val_rmse"])
                best = int(np.argmin(val))
                rows.append(dict(bias=bias, rank=k, lam=lam, best_iter=best + 1,
                                 val_rmse=round(float(val[best]), 4),
                                 val_rmse_final=round(float(val[-1]), 4),
                                 sec_per_iter=round(h["time"][-1] / N_ITER, 3)))
                print(f"  {bias:<5} k={k:<3} lam={lam:<3} val={val[best]:.4f} "
                      f"(it. {best + 1}, finale {val[-1]:.4f})", flush=True)
    return rows


def convergence_curves(R_fit, R_val, rank, lam):
    """GD et ALS sur la même config : historique complet pour comparer les vitesses."""
    runs = {
        "GD": MatrixFactorizationGD(rank=rank, lam=lam, n_epochs=800, lr=3e-4, momentum=0.9),
        "ALS": MatrixFactorizationALS(rank=rank, lam=lam, n_epochs=N_ITER),
    }
    rows = []
    for name, model in runs.items():
        h = model.fit(R_fit, R_val).history
        rows += [dict(method=name, epoch=e, time=round(t, 3), loss=round(l, 1), val_rmse=round(v, 4))
                 for e, t, l, v in zip(h["epoch"], h["time"], h["loss"], h["val_rmse"])]
        val = np.array(h["val_rmse"])
        best = int(np.argmin(val))
        print(f"  {name:<4} meilleure val {val[best]:.4f} à l'époque {best + 1} "
              f"({h['time'][best]:.1f}s) | coût final {h['loss'][-1]:.0f} ({h['time'][-1]:.1f}s)")
    return rows


def main():
    R_train, R_test = load_train_test()
    R_fit, R_val = split_observed(R_train, frac=0.1, seed=0)
    RESULTS.mkdir(exist_ok=True)

    print("Grille ALS sur la validation :")
    rows = grid_search(R_fit, R_val)
    write_csv(RESULTS / "als_grid.csv", rows)

    best_full = min((r for r in rows if r["bias"] == "full"), key=lambda r: r["val_rmse"])
    print(f"\nGD vs ALS a configuration egale (full, k={best_full['rank']}, lam={best_full['lam']}) :")
    write_csv(RESULTS / "gd_vs_als_curves.csv",
              convergence_curves(R_fit, R_val, best_full["rank"], best_full["lam"]))

    models = {
        "Baseline biais regularises (lam=2)": Biases(lam_u=2, lam_i=2),
        f"MF-GD full (k={GD_BEST['rank']}, lam={GD_BEST['lam']}, {GD_BEST['n_epochs']} ep.)":
            MatrixFactorizationGD(**GD_BEST),
    }
    for bias in ["none", "mean", "full"]:
        best = min((r for r in rows if r["bias"] == bias), key=lambda r: r["val_rmse"])
        name = f"MF-ALS {bias:<4} (k={best['rank']}, lam={best['lam']}, {best['best_iter']} it.)"
        models[name] = MatrixFactorizationALS(rank=best["rank"], lam=best["lam"], bias=bias,
                                              n_epochs=best["best_iter"], init_std=INIT_STD[bias])

    print(f"\n{'Modele':<45}{'RMSE train':>11}{'RMSE test':>11}{'Acc test':>10}{'Temps':>9}")
    for name, model in models.items():
        t0 = time.perf_counter()
        P = model.fit(R_train).predict()
        dt = time.perf_counter() - t0
        print(f"{name:<45}{rmse(R_train, P):>11.4f}{rmse(R_test, P):>11.4f}"
              f"{accuracy(R_test, P):>10.3f}{dt:>8.2f}s")


if __name__ == "__main__":
    main()
