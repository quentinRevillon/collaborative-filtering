"""Factorisation de matrice par descente de gradient, sur le split local train -> test.

Trois variantes, du modèle du cours vers le modèle complet :
    none : R ≈ U I^T            mean : R ≈ mu + U I^T            full : R ≈ mu + b_u + b_i + U I^T

Protocole (le même que src/run_baselines.py) :
  1. grille (rang k, lambda) sur une validation = 10 % des notes de train ; pour chaque
     config on garde la meilleure époque (early stopping) ;
  2. la meilleure config de chaque variante est réentraînée sur tout le train avec ce
     nombre d'époques, puis évaluée sur le test.
Les résultats de la grille sont écrits dans results/mf_gd_grid.csv.

Lancer depuis la racine du repo :  python -m src.run_mf_gd
"""
import csv
import time
from pathlib import Path

import numpy as np

from src.baselines import Biases
from src.data import load_train_test, split_observed
from src.metrics import accuracy, rmse
from src.mf import MatrixFactorizationGD

RANKS = [2, 10, 50]
LAMBDAS = [1, 5, 10, 15, 20]
N_EPOCHS = 800
GD = dict(lr=3e-4, momentum=0.9)
# Le modèle pur doit représenter toute la note (~3.5) avec <u_u, i_i> : init plus large.
INIT_STD = {"none": 0.5, "mean": 0.1, "full": 0.1}
RESULTS = Path(__file__).resolve().parent.parent / "results"


def grid_search(R_fit, R_val):
    rows = []
    for bias in ["none", "mean", "full"]:
        for k in RANKS:
            for lam in LAMBDAS:
                model = MatrixFactorizationGD(rank=k, lam=lam, n_epochs=N_EPOCHS, bias=bias,
                                              init_std=INIT_STD[bias], **GD)
                val = np.array(model.fit(R_fit, R_val).history["val_rmse"])
                best = int(np.argmin(val))
                rows.append(dict(bias=bias, rank=k, lam=lam, best_epoch=best + 1,
                                 val_rmse=round(float(val[best]), 4),
                                 val_rmse_final=round(float(val[-1]), 4)))
                print(f"  {bias:<5} k={k:<3} lam={lam:<3} val={val[best]:.4f} (époque {best + 1})",
                      flush=True)
    return rows


def main():
    R_train, R_test = load_train_test()
    R_fit, R_val = split_observed(R_train, frac=0.1, seed=0)

    print("Grille sur la validation :")
    rows = grid_search(R_fit, R_val)
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "mf_gd_grid.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    models = {"Baseline biais regularises (lam=2)": Biases(lam_u=2, lam_i=2)}
    for bias in ["none", "mean", "full"]:
        best = min((r for r in rows if r["bias"] == bias), key=lambda r: r["val_rmse"])
        name = f"MF-GD {bias:<4} (k={best['rank']}, lam={best['lam']}, {best['best_epoch']} ep.)"
        models[name] = MatrixFactorizationGD(rank=best["rank"], lam=best["lam"], bias=bias,
                                             n_epochs=best["best_epoch"],
                                             init_std=INIT_STD[bias], **GD)

    print(f"\n{'Modele':<45}{'RMSE train':>11}{'RMSE test':>11}{'Acc test':>10}{'Temps':>9}")
    for name, model in models.items():
        t0 = time.perf_counter()
        P = model.fit(R_train).predict()
        dt = time.perf_counter() - t0
        print(f"{name:<45}{rmse(R_train, P):>11.4f}{rmse(R_test, P):>11.4f}"
              f"{accuracy(R_test, P):>10.3f}{dt:>8.2f}s")


if __name__ == "__main__":
    main()
