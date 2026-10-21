"""Protocole unique pour le rapport : tous les modèles sur les 5 mêmes plis de train + test.

Les hyperparamètres sont ceux déjà choisis par CV 5 plis sur le train seul (expériences
baselines, mf_gd, mf_optuna, knn, koren, bpmf_alpha) ; seuls le rang et le nombre d'itérations
de la Bayesian FM ont été choisis sur train + test (bfm_platform). Chaque modèle est ajusté sur
4 plis avec son nombre d'itérations fixé, puis scoré sur le 5e : aucune sélection sur les plis
d'évaluation, à part celle de la FM.

Pour chaque modèle : RMSE par pli, moyenne, temps d'ajustement + prédiction par pli, et
comparaison appariée avec un modèle de référence : nombre de plis gagnés et t-test corrigé
de Nadeau et Bengio (2003), qui tient compte du recouvrement des plis d'apprentissage :
    t = mean(d) / sqrt((1/K + n_val/n_fit) var(d)),  K - 1 degrés de liberté.

Sortie : results/unified_cv.csv (une ligne par modèle, complétée au fur et à mesure),
results/unified_cv_histories.npz (courbes de Gibbs de BPMF et de la FM complète).

Lancer depuis la racine du repo :  python -m src.experiments.unified_cv [--only biases bpmf_15 ...]
"""
import argparse
import time

import numpy as np
from scipy import stats

from src.data import DATA_DIR, kfold_observed, load_train_test
from src.evaluation import RESULTS_DIR, read_csv, write_csv
from src.metrics import rmse
from src.models import (BPMF, BayesianFM, Biases, GlobalMean, KNNResidual, KorenNeighborhood,
                        MatrixFactorizationALS, MatrixFactorizationGD)

GD = dict(lr=3e-4, momentum=0.9)
FM = dict(rank=20, n_epochs=238, burn_in=20, seed=0)
BPMF_BASE = dict(bias="full", rank=10, burn_in=20, seed=0)

# nom -> (description, constructeur, référence pour la comparaison appariée)
CONFIGS = {
    "global_mean": ("Global mean", lambda: GlobalMean(), None),
    "biases_noreg": ("Biases, no regularization", lambda: Biases(n_iter=20), None),
    "biases": ("Biases, regularized", lambda: Biases(lam_u=5, lam_i=3), "biases_noreg"),
    "mf_raw": ("MF on raw ratings, GD (k=2, lam=1)", lambda: MatrixFactorizationGD(
        bias="none", rank=2, lam=1, n_epochs=90, init_std=0.5, **GD), None),
    "mf_gd": ("MF with biases, GD (k=50, lam=10)", lambda: MatrixFactorizationGD(
        bias="full", rank=50, lam=10, n_epochs=244, **GD), "biases"),
    "mf_als": ("MF with biases, ALS + Optuna (k=2)", lambda: MatrixFactorizationALS(
        bias="full", rank=2, lam_u=53.91, lam_i=9.221, lam_bias=3.41, n_epochs=25), "biases"),
    "knn_user": ("User kNN (k=40)", lambda: KNNResidual(
        kind="user", k=40, shrink=50, damp=2, lam_u=5, lam_i=3), "biases"),
    "knn_item": ("Item kNN (k=20)", lambda: KNNResidual(
        kind="item", k=20, shrink=100, damp=0.5, lam_u=5, lam_i=3), "biases"),
    "koren": ("Koren, learnt weights (k=200)", lambda: KorenNeighborhood(
        k=200, lam_w=3, lam_u=5, lam_i=3, shrink=100, n_epochs=10), "biases"),
    "bpmf_2": ("BPMF, alpha = 2 (paper)", lambda: BPMF(alpha=2.0, n_epochs=119, **BPMF_BASE), "biases"),
    "bpmf_15": ("BPMF, alpha = 1.5 (CV)", lambda: BPMF(alpha=1.5, n_epochs=133, **BPMF_BASE), "mf_gd"),
    "bpmf_gibbs": ("BPMF, alpha sampled", lambda: BPMF(alpha="gibbs", n_epochs=148, **BPMF_BASE), "bpmf_15"),
    "fm_none": ("FM, no features (= BPMF)", lambda: BayesianFM(features=(), **FM), None),
    "fm_implicit": ("FM + implicit feedback", lambda: BayesianFM(features=("iu", "ii"), **FM), "fm_none"),
    "fm_genres": ("FM + genres", lambda: BayesianFM(features=("side",), **FM), "fm_none"),
    "fm_all": ("FM + implicit feedback + genres", lambda: BayesianFM(
        features=("iu", "ii", "side"), **FM), "fm_genres"),
    "fm_all_oldgenres": ("FM + implicit + first genre file", lambda: BayesianFM(
        features=("iu", "ii", "side"), genres_path=DATA_DIR / "namesngenre.npy", **FM), None),
}
HISTORY = {"bpmf_15", "bpmf_gibbs", "fm_all"}   # modèles dont on garde les courbes de Gibbs


def corrected_ttest(d, ratio):
    """t-test apparié corrigé (Nadeau et Bengio) sur les différences par pli d ; ratio = n_val / n_fit."""
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return float("nan")
    t = d.mean() / np.sqrt((1 / k + ratio) * var)
    return float(2 * stats.t.sf(abs(t), df=k - 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", default=list(CONFIGS))
    args = parser.parse_args()

    R_train, R_test = load_train_test()
    R_all = np.where(np.isnan(R_train), R_test, R_train)
    folds = kfold_observed(R_all, n_folds=5)
    n_fit = (~np.isnan(folds[0][0])).sum()
    ratio = (~np.isnan(folds[0][1])).sum() / n_fit
    print(f"train + test : {int((~np.isnan(R_all)).sum())} notes, ~{n_fit} par pli d'apprentissage", flush=True)

    rows = {r["model"]: r for r in read_csv("unified_cv.csv")} if (RESULTS_DIR / "unified_cv.csv").exists() else {}
    histories = dict(np.load(RESULTS_DIR / "unified_cv_histories.npz")) \
        if (RESULTS_DIR / "unified_cv_histories.npz").exists() else {}

    for name in args.only:
        label, make, ref = CONFIGS[name]
        scores, times, hist = [], [], []
        for R_fit, R_val in folds:
            t0 = time.perf_counter()
            model = make()
            if name in HISTORY:
                model.fit(R_fit, R_val)
                hist.append(model.history)
            else:
                model.fit(R_fit)
            scores.append(rmse(R_val, model.predict()))
            times.append(time.perf_counter() - t0)
        scores = np.array(scores)
        row = dict(model=name, label=label, cv_rmse=round(float(scores.mean()), 4),
                   cv_std=round(float(scores.std()), 4), seconds_per_fold=round(float(np.mean(times)), 1),
                   per_fold=" ".join(f"{s:.4f}" for s in scores))
        if ref is not None and ref in rows:
            d = scores - np.array([float(v) for v in str(rows[ref]["per_fold"]).split()])
            row.update(ref=ref, gain=round(float(d.mean()), 4), folds_better=int((d < 0).sum()),
                       p_value=round(corrected_ttest(d, ratio), 4))
        rows[name] = row
        for key in ("val_rmse", "val_rmse_sample", "alpha"):
            if hist and key in hist[0] and len(hist[0][key]):
                histories[f"{name}_{key}"] = np.array([h[key] for h in hist])
        print(f"  {label:<38} {row['cv_rmse']:.4f} +/- {row['cv_std']:.4f}  {row['seconds_per_fold']:>6.1f} s/pli"
              + (f"  vs {ref}: {row['gain']:+.4f}, {row['folds_better']}/5, p={row['p_value']}" if "gain" in row else ""),
              flush=True)
        write_csv("unified_cv.csv", [rows[k] for k in CONFIGS if k in rows])
        if histories:
            np.savez(RESULTS_DIR / "unified_cv_histories.npz", **histories)


if __name__ == "__main__":
    main()
