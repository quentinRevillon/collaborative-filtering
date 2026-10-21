"""Stack final : Bayesian FM sur notes normalisées par user (plusieurs chaînes), FM brute, kNN
films, puis gradient boosting sur le résidu. CV 5 plis sur train + test, mêmes plis que
src/experiments/bfm_stacking.py et bfm_user_norm.py.

1. Prédictions hors pli de la FM normalisée (centrage + réduction, k = 20, rang 20, 238
   itérations) pour --chains graines ; la graine 0 est reprise de bfm_user_norm_oof.npz, les
   autres sont mises en cache dans results/final_stack_norm_oof.npz (--reuse).
2. Comparaison, pli par pli, à la soumission actuelle (stack linéaire FM brute 5 chaînes + kNN) :
   FM normalisée seule, stacks linéaires, et gradient boosting sur le résidu du stack linéaire
   (poids et boosting appris sur 4 plis, évalués sur le 5e).
3. Écrit les données nécessaires à la soumission : prédictions hors pli et caractéristiques de
   chaque note (results/final_stack_train.npz), pour réentraîner le boosting dans generate.py.
Sorties : results/final_stack.csv, results/final_stack_train.npz.

Lancer depuis la racine du repo :  python -m src.experiments.final_stack [--chains 5] [--reuse]
"""
import argparse
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from src.data import kfold_observed, load_train_test
from src.evaluation import RESULTS_DIR, write_csv
from src.experiments.bfm_stacking import FM
from src.experiments.error_analysis import load_oof, movie_meta
from src.models.bfm import BayesianFM
from src.models.normalized import UserNormalized

NORM = dict(shrink=20.0, center=True, scale=True)
GBDT = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.0,
            early_stopping=False, random_state=0)
FEATURE_NAMES = ["fm_norm", "fm_raw", "knn", "chain_std", "log_n_i", "log_n_u", "std_u", "mean_u", "half_u",
                 "mean_i", "std_i", "decade"]


def norm_chains(folds, n_chains, reuse):
    """Prédictions hors pli de la FM normalisée, une colonne par graine."""
    cache_path = RESULTS_DIR / "final_stack_norm_oof.npz"
    cache = dict(np.load(cache_path)) if reuse and cache_path.exists() else {}
    cache.setdefault("seed_0", np.load(RESULTS_DIR / "bfm_user_norm_oof.npz")["v1"])   # k = 20, graine 0
    for s in range(n_chains):
        if f"seed_{s}" in cache:
            continue
        parts = []
        for k, (R_fit, R_val) in enumerate(folds):
            t0 = time.perf_counter()
            vu, vi = np.nonzero(~np.isnan(R_val))
            model = UserNormalized(BayesianFM(**FM, seed=s, clip=False), clip=False, **NORM).fit(R_fit)
            parts.append(model.predict()[vu, vi])
            print(f"  FM normalisee, graine {s}, pli {k + 1} ({time.perf_counter() - t0:.0f} s)", flush=True)
        cache[f"seed_{s}"] = np.concatenate(parts)
        np.savez(cache_path, **cache)
    return np.array([cache[f"seed_{s}"] for s in range(n_chains)])


def linear(X, y, fold):
    pred, base_tr = np.empty_like(y), {}
    for k in range(fold.max() + 1):
        tr, te = fold != k, fold == k
        A = np.column_stack([np.ones(tr.sum()), X[tr]])
        w = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        pred[te] = w[0] + X[te] @ w[1:]
        base_tr[k] = w[0] + X[tr] @ w[1:]
    return pred, base_tr


def boosted(X, feats, y, fold):
    """Stack linéaire + gradient boosting sur son résidu, appris sur 4 plis, évalué sur le 5e."""
    lin, base_tr = linear(X, y, fold)
    pred = np.empty_like(y)
    for k in range(fold.max() + 1):
        tr, te = fold != k, fold == k
        gb = HistGradientBoostingRegressor(**GBDT).fit(feats[tr], y[tr] - base_tr[k])
        pred[te] = lin[te] + gb.predict(feats[te])
    return pred


def per_fold(pred, y, fold):
    p = np.clip(pred, 0.5, 5.0)
    return np.array([np.sqrt(np.mean((y[fold == k] - p[fold == k]) ** 2)) for k in range(fold.max() + 1)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chains", type=int, default=5)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()

    oof = load_oof()                       # FM brute (10 graines), kNN, statistiques par pli
    y, fold = oof["truth"], oof["fold_id"]
    R_train, R_test = load_train_test()
    folds = kfold_observed(np.where(np.isnan(R_train), R_test, R_train), n_folds=5)
    print(f"Predictions hors pli de la FM normalisee ({args.chains} chaines) :", flush=True)
    norm = norm_chains(folds, args.chains, args.reuse)
    raw = np.array([oof[f"fm_{s}"] for s in range(5)])
    fm_norm, fm_raw, knn = norm.mean(0), raw.mean(0), oof["knn_films"]

    _, genres, decades = movie_meta()
    names = sorted({g for gs in genres for g in gs})
    G = np.array([[g in gs for g in names] for gs in genres], dtype=float)
    feats = np.column_stack([fm_norm, fm_raw, knn, norm.std(0), np.log1p(oof["n_i"]), np.log1p(oof["n_u"]),
                             oof["std_u"], oof["mean_u"], oof["half_u"], oof["mean_i"], oof["std_i"],
                             decades[oof["item"]], G[oof["item"]]])

    current = linear(np.column_stack([fm_raw, knn]), y, fold)[0]
    ref = per_fold(current, y, fold)
    variants = {
        "soumission actuelle : FM brute (5) + kNN": current,
        "FM normalisee (1 chaine)": norm[0],
        f"FM normalisee ({args.chains} chaines)": fm_norm,
        f"lineaire : FM normalisee ({args.chains}) + kNN": linear(np.column_stack([fm_norm, knn]), y, fold)[0],
        f"lineaire : FM normalisee ({args.chains}) + FM brute (5) + kNN":
            linear(np.column_stack([fm_norm, fm_raw, knn]), y, fold)[0],
        "boosting sur FM normalisee + kNN": boosted(np.column_stack([fm_norm, knn]), feats, y, fold),
        "boosting sur FM normalisee + FM brute + kNN": boosted(np.column_stack([fm_norm, fm_raw, knn]), feats, y, fold),
    }
    rows = []
    print("\nComparaison a la soumission actuelle (CV 5 plis sur train + test) :")
    for name, pred in variants.items():
        s = per_fold(pred, y, fold)
        d = s - ref
        rows.append(dict(variant=name, cv_rmse=round(s.mean(), 4), diff_vs_submission=round(d.mean(), 5),
                         folds_better=int((d < 0).sum()), per_fold_diff=" ".join(f"{v:+.4f}" for v in d)))
        print(f"  {name:<58} cv={s.mean():.4f}  ecart {d.mean():+.5f}  ({' '.join(f'{v:+.4f}' for v in d)})",
              flush=True)
    write_csv("final_stack.csv", rows)
    np.savez_compressed(RESULTS_DIR / "final_stack_train.npz", user=oof["user"], item=oof["item"], truth=y,
                        fold_id=fold, fm_norm_chains=norm.astype(np.float32), fm_raw_chains=raw.astype(np.float32),
                        knn=knn, n_i=oof["n_i"], n_u=oof["n_u"], std_u=oof["std_u"], mean_u=oof["mean_u"],
                        half_u=oof["half_u"], mean_i=oof["mean_i"], std_i=oof["std_i"])


if __name__ == "__main__":
    main()
