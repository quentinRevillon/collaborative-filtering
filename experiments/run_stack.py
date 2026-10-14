"""Select the final stacked ensemble on validation, learn its weights, score it once on test.

Usage:
    python experiments/run_stack.py

Inputs: cached member predictions written by experiments/run_ensemble.py
(results/ensemble_preds.npz) and experiments/run_residual_models.py
(results/preds/res_fm_gbdt_*.npy). For every member, "valid" predictions come from the
member fitted on the 90 % part of train (split seed 0) and "test" predictions from the
member refitted on all of train.

Protocol: each candidate combination is scored by 5-fold CV RMSE of a ridge regression
(alpha = 10) on the 3,159 validation ratings; the best one is selected on that score only.
Its weights are then fitted on all validation ratings and it is scored once on test.
The weights are frozen in src/stack.py (WEIGHTS), so generate.py fits each member once.
Writes results/stack_candidates.csv, results/stack_weights.csv, results/stack_results.csv.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run as a script from the repo root
from src.data import accuracy, clip, load_coo, rmse, split

ALPHA = 10.0
CANDIDATES = {
    "fm": ("fm",),
    "fm+knn+fm_iu_ii": ("fm", "knn", "fm_iu_ii"),
    "fm_gbdt": ("fm_gbdt",),
    "fm_gbdt+knn": ("fm_gbdt", "knn"),
    "fm+knn+fm_iu_ii+fm_gbdt": ("fm", "knn", "fm_iu_ii", "fm_gbdt"),
}


def load_preds(part):
    """{member: predictions} on `part` ("valid" or "test") from the cached files."""
    ens = np.load("results/ensemble_preds.npz")
    preds = {name[len(part) + 1:]: ens[name] for name in ens.files if name.startswith(part)}
    preds["fm_gbdt"] = np.load(f"results/preds/res_fm_gbdt_{part}.npy")
    return preds


def design(preds, names):
    """(n, len(names)) matrix of the members' predictions."""
    return np.column_stack([preds[n] for n in names])


def cv_score(preds, names, r):
    """5-fold CV RMSE of the ridge stack of `names` on the ratings r."""
    return rmse(clip(cross_val_predict(Ridge(alpha=ALPHA), design(preds, names), r, cv=5)), r)


def save_csv(path, rows):
    """Write a list of dicts as CSV."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    valid = split(load_coo("data/ratings_train.npy")[0], 0.1, 0)[1]
    test = load_coo("data/ratings_test.npy")[0]
    pv, pt = load_preds("valid"), load_preds("test")
    scores = {k: cv_score(pv, names, valid[2]) for k, names in CANDIDATES.items()}
    save_csv("results/stack_candidates.csv", [dict(stack=k, valid_cv_rmse=round(s, 5)) for k, s in scores.items()])
    best = min(scores, key=scores.get)
    ridge = Ridge(alpha=ALPHA).fit(design(pv, CANDIDATES[best]), valid[2])
    pred = clip(ridge.predict(design(pt, CANDIDATES[best])))
    save_csv("results/stack_weights.csv", [dict(input=n, weight=round(w, 4)) for n, w in
                                           zip(CANDIDATES[best] + ("intercept",), list(ridge.coef_) + [ridge.intercept_])])
    save_csv("results/stack_results.csv", [dict(stack=best, valid_cv_rmse=round(scores[best], 5),
                                                test_rmse=round(rmse(pred, test[2]), 4), test_acc=round(accuracy(pred, test[2]), 4))])
    print(scores, best, rmse(pred, test[2]))
