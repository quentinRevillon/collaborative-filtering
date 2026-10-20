"""Vérification de la Bayesian FM de main (src/bfm.py, src/side_info.py) : ancien fichier de
genres (namesngenre.npy, mal aligné) vs fichier corrigé (namesngenre2.npy).

1. CV 5 plis sur ratings_train (nos plis), rang 10, 1000 itérations : sans genres, genres
   seuls et modèle complet (iu+ii+side), avec l'ancien puis le nouveau fichier ;
2. train -> test de la config finale d'origine de main (complet, rang 10, 2000 itérations).
Sorties : results/main_bfm_genres.csv (+ log console).

Lancer depuis la racine du repo :  python -m src.experiments.main_bfm_genres [--main ../a1-main]
"""
import time

import numpy as np

from src.data import DATA_DIR, kfold_observed, load_train_test
from src.evaluation import write_csv
from src.experiments.main_code import coo, load_main_module, main_path_from_args

GENRE_FILES = {"ancien": DATA_DIR / "namesngenre.npy", "nouveau": DATA_DIR / "namesngenre2.npy"}
CONFIGS = {  # nom -> (features, fichier de genres)
    "sans genres (iu+ii)": (("iu", "ii"), "nouveau"),
    "genres seuls, ancien": (("side",), "ancien"),
    "genres seuls, nouveau": (("side",), "nouveau"),
    "complet (iu+ii+side), ancien": (("iu", "ii", "side"), "ancien"),
    "complet (iu+ii+side), nouveau": (("iu", "ii", "side"), "nouveau"),
}


def main():
    args = main_path_from_args(__doc__.splitlines()[0])
    bfm = load_main_module(args.main, "bfm")
    side = load_main_module(args.main, "side_info")
    F = {name: side.item_features(path) for name, path in GENRE_FILES.items()}
    print("features : " + " | ".join(f"{n} {f.shape}" for n, f in F.items()), flush=True)

    def score(R_fit, R_eval, features, genres, n_iter):
        t0 = time.perf_counter()
        predict = bfm.fit_bfm(coo(R_fit), R_fit.shape, F[genres], features=features, rank=10,
                              n_iter=n_iter)
        u, i, r = coo(R_eval)
        pred = np.clip(predict(u, i), 0.5, 5.0)
        return float(np.sqrt(np.mean((r - pred) ** 2))), time.perf_counter() - t0

    R_train, R_test = load_train_test()
    folds = kfold_observed(R_train, n_folds=args.folds)
    cv = {name: [] for name in CONFIGS}
    for k, (R_fit, R_val) in enumerate(folds, 1):
        for name, (features, genres) in CONFIGS.items():
            cv[name].append(score(R_fit, R_val, features, genres, n_iter=1000)[0])
        print(f"pli {k} : " + "  ".join(f"{n}={cv[n][-1]:.4f}" for n in CONFIGS), flush=True)

    rows = []
    print(f"\nCV {args.folds} plis (rang 10, 1000 iterations)")
    for name, (features, genres) in CONFIGS.items():
        s = np.array(cv[name])
        rows.append(dict(eval="cv5", config=name, features="+".join(features), genres=genres,
                         n_iter=1000, rmse=round(s.mean(), 4), std=round(s.std(), 4),
                         per_fold=" ".join(f"{v:.4f}" for v in s)))
        print(f"  {name:<32} {s.mean():.4f} +/- {s.std():.4f}")
    d = np.array(cv["complet (iu+ii+side), nouveau"]) - np.array(cv["complet (iu+ii+side), ancien"])
    print(f"  complet nouveau - ancien, par pli : {' '.join(f'{v:+.4f}' for v in d)}  (moyenne {d.mean():+.4f})")

    print("\nTrain -> test, config finale d'origine de main (iu+ii+side, rang 10, 2000 iterations) :")
    for genres in GENRE_FILES:
        rmse, dt = score(R_train, R_test, ("iu", "ii", "side"), genres, n_iter=2000)
        rows.append(dict(eval="test", config=f"complet, {genres}", features="iu+ii+side", genres=genres,
                         n_iter=2000, rmse=round(rmse, 4), std=None, per_fold=None))
        print(f"  genres {genres:<8} RMSE test {rmse:.4f}  ({dt:.0f} s)", flush=True)
    write_csv("main_bfm_genres.csv", rows)


if __name__ == "__main__":
    main()
