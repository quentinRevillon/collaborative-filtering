"""Bayesian FM (rang 20, 238 itérations) : moyennage de plusieurs chaînes et stacking avec
des kNN, en CV 5 plis sur train + test (taille de la plateforme).

1. Prédictions hors pli : dans chaque pli, les modèles de base sont entraînés sur les 80 %
   d'apprentissage et prédisent les 20 % de validation. Chaque note de train + test reçoit ainsi
   une prédiction de modèles qui ne l'ont pas vue. Modèles : Bayesian FM (--chains graines),
   kNN films et kNN users sur les résidus, biais régularisés.
2. Moyennage de chaînes : moyenne des prédictions des n premières graines.
3. Stacking : combinaison linéaire (moindres carrés + intercept) des prédictions hors pli ; les
   poids sont appris sur 4 plis et évalués sur le 5e (à tour de rôle), puis réappris sur tous
   les plis pour la soumission.
Chaque variante est comparée, pli par pli, à la Bayesian FM à une seule chaîne.
Sorties : results/bfm_stacking.csv, results/bfm_stacking_weights.csv, results/bfm_stacking_oof.npz.

Lancer depuis la racine du repo :  python -m src.experiments.bfm_stacking [--folds 5] [--chains 5] [--reuse]
(--reuse : reprend les prédictions déjà présentes dans bfm_stacking_oof.npz, mêmes plis,
 et ne calcule que les chaînes manquantes)
"""
import argparse
import time

import numpy as np

from src.data import kfold_observed, load_train_test
from src.evaluation import RESULTS_DIR, write_csv
from src.models import Biases, KNNResidual
from src.models.bfm import BayesianFM

FM = dict(rank=20, n_epochs=238, burn_in=20, features=("iu", "ii", "side"))
OTHERS = {
    "knn_films": lambda: KNNResidual(kind="item", k=20, shrink=100, damp=0.5, clip=False),
    "knn_users": lambda: KNNResidual(kind="user", k=40, shrink=50, damp=2, clip=False),
    "biais": lambda: Biases(lam_u=5, lam_i=3, clip=False),
}


def out_of_fold(folds, n_chains, known=None):
    """Prédictions hors pli de chaque modèle de base, pour toutes les notes de validation.
    `known` : prédictions hors pli déjà calculées sur les mêmes plis (réutilisées telles quelles)."""
    known = known or {}
    preds, truth, fold_id = {}, [], []
    for k, (R_fit, R_val) in enumerate(folds):
        vu, vi = np.nonzero(~np.isnan(R_val))
        truth.append(R_val[vu, vi])
        fold_id.append(np.full(len(vu), k))
        for s in range(n_chains):
            if f"fm_{s}" in known:
                continue
            t0 = time.perf_counter()
            P = BayesianFM(**FM, seed=s, clip=False).fit(R_fit).predict()
            preds.setdefault(f"fm_{s}", []).append(P[vu, vi])
            print(f"  pli {k + 1} : chaine FM {s} ({time.perf_counter() - t0:.0f} s)", flush=True)
        for name, make in OTHERS.items():
            if name not in known:
                preds.setdefault(name, []).append(make().fit(R_fit).predict()[vu, vi])
    preds = {n: np.concatenate(v) for n, v in preds.items()}
    return {**known, **preds}, np.concatenate(truth), np.concatenate(fold_id)


def fit_blend(X, y):
    """Poids (intercept en premier) des moindres carrés y ~ 1 + X."""
    A = np.column_stack([np.ones(len(y)), X])
    return np.linalg.lstsq(A, y, rcond=None)[0]


def blend(X, w):
    return np.clip(w[0] + X @ w[1:], 0.5, 5.0)


def per_fold_rmse(pred, truth, fold_id, n_folds):
    return np.array([np.sqrt(np.mean((truth[fold_id == k] - pred[fold_id == k]) ** 2))
                     for k in range(n_folds)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--chains", type=int, default=5)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()

    R_train, R_test = load_train_test()
    R_all = np.where(np.isnan(R_train), R_test, R_train)
    folds = kfold_observed(R_all, n_folds=args.folds)
    print(f"Predictions hors pli, CV {args.folds} plis sur train + test, {args.chains} chaines FM :")
    known = None
    if args.reuse and (RESULTS_DIR / "bfm_stacking_oof.npz").exists():
        saved = np.load(RESULTS_DIR / "bfm_stacking_oof.npz")
        known = {k: saved[k] for k in saved.files if k not in ("truth", "fold_id")}
        print(f"  reprise de {sorted(known)}", flush=True)
    preds, truth, fold_id = out_of_fold(folds, args.chains, known)
    if known:
        assert np.array_equal(truth, saved["truth"]), "plis différents de ceux de la sauvegarde"
    np.savez(RESULTS_DIR / "bfm_stacking_oof.npz", truth=truth, fold_id=fold_id, **preds)

    def fm_avg(n):
        return np.mean([preds[f"fm_{s}"] for s in range(n)], axis=0)

    variants = {f"FM, moyenne de {n} chaine(s)": ("avg", n, ())
                for n in sorted({1, 2, 3, 5, args.chains} & set(range(1, args.chains + 1)))}
    for n in sorted({1, 5, args.chains} & set(range(1, args.chains + 1))):
        variants[f"stack FM({n}) + kNN films"] = ("stack", n, ("knn_films",))
        variants[f"stack FM({n}) + kNN films + kNN users + biais"] = ("stack", n, tuple(OTHERS))

    rows, weights, ref = [], [], None
    for name, (kind, n, others) in variants.items():
        if kind == "avg":
            scores = per_fold_rmse(np.clip(fm_avg(n), 0.5, 5.0), truth, fold_id, args.folds)
        else:
            X = np.column_stack([fm_avg(n)] + [preds[o] for o in others])
            pred = np.empty_like(truth)
            for k in range(args.folds):   # poids appris sur les 4 autres plis
                w = fit_blend(X[fold_id != k], truth[fold_id != k])
                pred[fold_id == k] = blend(X[fold_id == k], w)
            scores = per_fold_rmse(pred, truth, fold_id, args.folds)
            w_all = fit_blend(X, truth)   # poids finaux, appris sur toutes les prédictions hors pli
            weights.append(dict(variant=name, intercept=round(w_all[0], 4),
                                **{f"w_{c}": round(v, 4) for c, v in zip([f"fm{n}"] + list(others), w_all[1:])}))
        if ref is None:
            ref = scores
        d = scores - ref
        rows.append(dict(variant=name, cv_rmse=round(scores.mean(), 4), cv_std=round(scores.std(), 4),
                         diff_vs_fm1=round(d.mean(), 4), folds_better=int((d < 0).sum()),
                         per_fold=" ".join(f"{v:.4f}" for v in scores)))
        print(f"  {name:<48} cv={scores.mean():.4f} +/- {scores.std():.4f}  ecart vs FM(1) {d.mean():+.4f}"
              f"  ({' '.join(f'{v:+.4f}' for v in d)})", flush=True)
    write_csv("bfm_stacking.csv", rows)
    write_csv("bfm_stacking_weights.csv", weights)
    print("\nPoids des stacks (appris sur toutes les predictions hors pli) :")
    for w in weights:
        print("  " + "  ".join(f"{k}={v}" for k, v in w.items()))


if __name__ == "__main__":
    main()
