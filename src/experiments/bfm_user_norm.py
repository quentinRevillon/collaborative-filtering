"""Bayesian FM sur notes normalisées par user (src/models/normalized.py), en CV 5 plis sur
train + test, mêmes plis que src/experiments/bfm_stacking.py.

Chaque variante est une seule chaîne (graine 0, rang 20, 238 itérations), comparée pli par pli à
la même chaîne sans normalisation (fm_0 de results/bfm_stacking_oof.npz). Pour la meilleure
variante : stack avec le kNN films, et stack à trois (FM brute + FM normalisée + kNN).
Sorties : results/bfm_user_norm.csv, results/bfm_user_norm_oof.npz.

Lancer depuis la racine du repo :  python -m src.experiments.bfm_user_norm
"""
import time

import numpy as np

from src.data import kfold_observed, load_train_test
from src.evaluation import RESULTS_DIR, write_csv
from src.experiments.bfm_stacking import FM
from src.models.bfm import BayesianFM
from src.models.normalized import UserNormalized

VARIANTS = {
    "centrage + reduction, k=5": dict(shrink=5.0, center=True, scale=True),
    "centrage + reduction, k=20": dict(shrink=20.0, center=True, scale=True),
    "reduction seule, k=5": dict(shrink=5.0, center=False, scale=True),
}


def per_fold(pred, y, fold):
    p = np.clip(pred, 0.5, 5.0)
    return np.array([np.sqrt(np.mean((y[fold == k] - p[fold == k]) ** 2)) for k in range(fold.max() + 1)])


def stack(X, y, fold):
    pred = np.empty_like(y)
    for k in range(fold.max() + 1):
        A = np.column_stack([np.ones((fold != k).sum()), X[fold != k]])
        w = np.linalg.lstsq(A, y[fold != k], rcond=None)[0]
        pred[fold == k] = w[0] + X[fold == k] @ w[1:]
    return pred


def report(name, pred, ref, y, fold, rows):
    s = per_fold(pred, y, fold)
    d = s - ref
    rows.append(dict(variant=name, cv_rmse=round(s.mean(), 4), diff=round(d.mean(), 5), folds_better=int((d < 0).sum()),
                     per_fold_diff=" ".join(f"{v:+.4f}" for v in d)))
    print(f"  {name:<52} cv={s.mean():.4f}  ecart {d.mean():+.5f}  ({' '.join(f'{v:+.4f}' for v in d)})", flush=True)


def main():
    saved = np.load(RESULTS_DIR / "bfm_stacking_oof.npz")
    y, fold = saved["truth"], saved["fold_id"]
    R_train, R_test = load_train_test()
    R_all = np.where(np.isnan(R_train), R_test, R_train)
    folds = kfold_observed(R_all, n_folds=5)

    oof = {name: [] for name in VARIANTS}
    for k, (R_fit, R_val) in enumerate(folds):
        vu, vi = np.nonzero(~np.isnan(R_val))
        assert np.array_equal(R_val[vu, vi], y[fold == k]), "plis differents de bfm_stacking_oof.npz"
        for name, params in VARIANTS.items():
            t0 = time.perf_counter()
            model = UserNormalized(BayesianFM(**FM, seed=0, clip=False), clip=False, **params).fit(R_fit)
            oof[name].append(model.predict()[vu, vi])
            print(f"  pli {k + 1} : {name} ({time.perf_counter() - t0:.0f} s)", flush=True)
    oof = {n: np.concatenate(v) for n, v in oof.items()}
    np.savez(RESULTS_DIR / "bfm_user_norm_oof.npz", truth=y, fold_id=fold,
             **{f"v{j}": oof[n] for j, n in enumerate(VARIANTS)})

    rows = []
    ref = per_fold(saved["fm_0"], y, fold)
    print("\nUne chaine (graine 0), comparee a la meme chaine sans normalisation :")
    report("sans normalisation (reference)", saved["fm_0"], ref, y, fold, rows)
    for name in VARIANTS:
        report(name, oof[name], ref, y, fold, rows)

    best = min(VARIANTS, key=lambda n: per_fold(oof[n], y, fold).mean())
    knn = saved["knn_films"]
    ref_stack = per_fold(stack(np.column_stack([saved["fm_0"], knn]), y, fold), y, fold)
    print(f"\nStacks avec la meilleure variante ({best}), compares au stack FM brute (1 chaine) + kNN :")
    report("FM brute + kNN (reference)", stack(np.column_stack([saved["fm_0"], knn]), y, fold), ref_stack, y, fold, rows)
    report("FM normalisee + kNN", stack(np.column_stack([oof[best], knn]), y, fold), ref_stack, y, fold, rows)
    report("FM brute + FM normalisee + kNN", stack(np.column_stack([saved["fm_0"], oof[best], knn]), y, fold),
           ref_stack, y, fold, rows)
    write_csv("bfm_user_norm.csv", rows)


if __name__ == "__main__":
    main()
