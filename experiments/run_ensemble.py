"""Ensembles (ridge / feature-weighted linear stacking / small GBM) around the Bayesian FM.

Usage:
    python experiments/run_ensemble.py            # reuses results/ensemble_preds.npz if present
    python experiments/run_ensemble.py --refit    # recompute every member's predictions

Protocol (test never used for any choice):
  1. train = data/ratings_train.npy, (fit, valid) = split(train, 0.1, seed=0).
  2. Every candidate member of src/ensemble.py is fitted on `fit` and predicts `valid`,
     then refitted on the full train and predicts the test pairs (cached).
  3. Every configuration (member subset x blender x alpha) is scored by the 5-fold CV RMSE
     of the blender on the valid ratings; the best one is selected on that score only.
  4. The selected blender is trained on all valid ratings and applied once to the test
     predictions of the members refitted on the full train (= src.ensemble.fit_ensemble(train)).
  5. Platform run: fit_ensemble on train+test merged and completion of the full matrix, timed.
Writes results/ensemble_members.csv, ensemble_configs.csv, ensemble_weights.csv,
ensemble_results.csv. Uses 3 CPU threads.
"""
import os
os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "3"
os.environ["TQDM_DISABLE"] = "1"

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run as a script from the repo root
from src.data import accuracy, clip, complete, load_coo, rmse, split
from src.ensemble import candidate_fits, counts, features, fit_ensemble, meta
from src.side_info import item_features

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
CACHE = RES / "ensemble_preds.npz"
THREADS = 3
ALPHAS = [0.1, 1.0, 10.0, 100.0]
TOL = 0.0005                        # parsimony: smallest config within TOL of the best CV RMSE
CLASSIC = ["fm", "bpmf", "knn", "side", "si", "als", "biases"]
FM_VARIANTS = ["fm", "fm_ii", "fm_iu_ii", "fm_mf", "fm_r5", "fm_r20"]
SUBSETS = {
    "fm": ["fm"],
    "fm+knn": ["fm", "knn"],
    "fm+knn+bpmf": ["fm", "knn", "bpmf"],
    "fm+knn+side": ["fm", "knn", "side"],
    "fm+knn+uknn": ["fm", "knn", "uknn"],
    "classic (7)": CLASSIC,
    "fm variants": FM_VARIANTS,
    "fm variants+knn": FM_VARIANTS + ["knn"],
    "fm variants+knn+bpmf": FM_VARIANTS + ["knn", "bpmf"],
    "all (14)": None,                       # filled with every member
}


def save_csv(rows, name):
    """Write a list of dicts as results/<name>."""
    with open(RES / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] {RES / name}")


def member_predictions(fit, tr, va, train, test, shape):
    """Clipped predictions of one member on va (fitted on tr) and on test (fitted on train), fit times."""
    t = time.perf_counter()
    pv = clip(fit(tr, shape)(va[0], va[1]))
    t1 = time.perf_counter()
    pt = clip(fit(train, shape)(test[0], test[1]))
    return pv, pt, t1 - t, time.perf_counter() - t1


def compute_cache(fits, tr, va, train, test, shape):
    """Predict valid / test with every member; save the arrays and the member table."""
    out, rows = {}, []
    for name, fit in fits.items():
        pv, pt, s1, s2 = member_predictions(fit, tr, va, train, test, shape)
        out[f"valid_{name}"], out[f"test_{name}"] = pv, pt
        rows.append(dict(member=name, valid_rmse=round(rmse(pv, va[2]), 4), fit_time_s=round(s1, 1),
                         fit_time_full_s=round(s2, 1)))
        print(f"  member {rows[-1]}", flush=True)
    np.savez(CACHE, **out)
    save_csv(rows, "ensemble_members.csv")


def blenders(alpha):
    """{blender name: (sklearn estimator, fwls flag)} for one ridge alpha."""
    return {"ridge": (Ridge(alpha=alpha), False), "fwls": (Ridge(alpha=alpha), True)}


def gbm():
    """Tiny gradient boosting on predictions + meta-features (shallow, strongly regularized)."""
    return HistGradientBoostingRegressor(max_depth=3, max_iter=150, learning_rate=0.05,
                                         min_samples_leaf=50, l2_regularization=1.0, random_state=0)


def cv_score(model, X, y):
    """5-fold CV RMSE of the clipped out-of-fold predictions of `model` on (X, y)."""
    return rmse(clip(cross_val_predict(model, X, y, cv=5)), y)


def design(cache, split_name, names, M, fwls):
    """Blend inputs of members `names` on 'valid' or 'test' from the cached predictions."""
    return features(np.column_stack([cache[f"{split_name}_{n}"] for n in names]), M, fwls)


def score_configs(cache, Mv, y):
    """CV RMSE of every (subset, blender, alpha) configuration and of the GBM; returns rows."""
    rows = []
    for subset, names in SUBSETS.items():
        for alpha in ALPHAS:
            for blender, (model, fwls) in blenders(alpha).items():
                rows.append(dict(subset=subset, members=" ".join(names), blender=blender, alpha=alpha,
                                 n_inputs=design(cache, "valid", names, Mv, fwls).shape[1],
                                 valid_cv_rmse=round(cv_score(model, design(cache, "valid", names, Mv, fwls), y), 5)))
        X = np.hstack([design(cache, "valid", names, Mv, False), Mv])
        rows.append(dict(subset=subset, members=" ".join(names), blender="gbm", alpha="",
                         n_inputs=X.shape[1], valid_cv_rmse=round(cv_score(gbm(), X, y), 5)))
        print(f"  {subset}: best {min(r['valid_cv_rmse'] for r in rows if r['subset'] == subset)}", flush=True)
    return rows


def greedy_selection(cache, Mv, y, pool, alpha):
    """Forward selection of members from 'fm' by ridge CV RMSE; returns the list of steps."""
    chosen, steps = ["fm"], []
    best = cv_score(Ridge(alpha=alpha), design(cache, "valid", chosen, Mv, False), y)
    steps.append((list(chosen), best))
    while len(chosen) < len(pool):
        scores = {n: cv_score(Ridge(alpha=alpha), design(cache, "valid", chosen + [n], Mv, False), y)
                  for n in pool if n not in chosen}
        n = min(scores, key=scores.get)
        if scores[n] >= best:
            break
        chosen, best = chosen + [n], scores[n]
        steps.append((list(chosen), best))
    return steps


def select(rows):
    """Best CV config, preferring the fewest inputs among those within TOL of the minimum."""
    lin = [r for r in rows if r["blender"] != "gbm"]
    m = min(r["valid_cv_rmse"] for r in rows)
    ok = [r for r in lin if r["valid_cv_rmse"] <= m + TOL]
    return min(ok, key=lambda r: (r["n_inputs"], r["valid_cv_rmse"])), m


def weight_rows(ridge, names, fwls):
    """Readable blend coefficients of a fitted ridge."""
    labels = list(names) + ([f"{n}*log_nu" for n in names] + [f"{n}*log_ni" for n in names]
                            + ["log_nu", "log_ni"] if fwls else [])
    return [dict(input=l, weight=round(float(w), 4)) for l, w in zip(labels, ridge.coef_)] + \
           [dict(input="intercept", weight=round(float(ridge.intercept_), 4))]


def platform_time(train, test, shape, F, best):
    """Seconds to fit the selected ensemble on train+test and complete the whole matrix."""
    both = tuple(np.concatenate(ab) for ab in zip(train, test))
    t = time.perf_counter()
    complete(fit_ensemble(both, shape, F, **best), shape)
    return time.perf_counter() - t


def main():
    """Cache member predictions, score blends on valid, test the selected one, time it."""
    torch.set_num_threads(THREADS)
    t0 = time.perf_counter()
    train, shape = load_coo(ROOT / "data" / "ratings_train.npy")
    test, _ = load_coo(ROOT / "data" / "ratings_test.npy")
    F = item_features(ROOT / "data" / "namesngenre.npy")
    tr, va = split(train, 0.1, seed=0)
    fits = candidate_fits(F)
    SUBSETS["all (14)"] = list(fits)
    if "--refit" in sys.argv or not CACHE.exists():
        compute_cache(fits, tr, va, train, test, shape)
    cache = np.load(CACHE)
    Mv, Mt = meta(counts(tr, shape), va[0], va[1]), meta(counts(train, shape), test[0], test[1])

    rows = score_configs(cache, Mv, va[2])
    for alpha in [1.0, 10.0]:
        for names, s in greedy_selection(cache, Mv, va[2], list(fits), alpha):
            rows.append(dict(subset=f"greedy step {len(names)} (a={alpha:g})", members=" ".join(names),
                             blender="ridge", alpha=alpha, n_inputs=len(names), valid_cv_rmse=round(s, 5)))
            print(f"  greedy a={alpha:g} {names}: {s:.5f}", flush=True)
    save_csv(rows, "ensemble_configs.csv")
    best, m = select(rows)
    names, fwls = best["members"].split(), best["blender"] == "fwls"
    print(f"selected {best} (best CV {m})", flush=True)

    ridge = Ridge(alpha=best["alpha"]).fit(design(cache, "valid", names, Mv, fwls), va[2])
    pred = clip(ridge.predict(design(cache, "test", names, Mt, fwls)))
    save_csv(weight_rows(ridge, names, fwls), "ensemble_weights.csv")
    call = dict(names=tuple(names), alpha=best["alpha"], fwls=fwls)
    sec = platform_time(train, test, shape, F, call)
    result = dict(method="ensemble", call=json.dumps(call), valid_cv_rmse=best["valid_cv_rmse"],
                  test_rmse=round(rmse(pred, test[2]), 4), test_acc=round(accuracy(pred, test[2]), 4),
                  fm_test_rmse=round(rmse(cache["test_fm"], test[2]), 4),
                  fm_valid_rmse=round(rmse(cache["valid_fm"], va[2]), 4),
                  threads=THREADS, train_test_fit_complete_s=round(sec, 1))
    print(result, flush=True)
    save_csv([result], "ensemble_results.csv")
    print(f"total time {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
