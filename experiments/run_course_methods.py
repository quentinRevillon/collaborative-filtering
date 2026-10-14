"""The course methods only: mean + biases, kNN, matrix factorization (with / without baseline).

Usage:
    python experiments/run_course_methods.py

Protocol: every setting is fitted on 90 % of the train ratings and scored on the other
10 % (validation, seed 0). For each variant, the best setting on validation is refitted on
all train ratings and scored once on test. Writes results/course_grid.csv (every setting)
and results/course_results.csv (best setting of each variant, with train / valid / test RMSE).

Variants:
  * biases: global mean; mean + user biases; mean + movie biases; both, lambda = 0 / tuned;
  * kNN: course formula on raw ratings (user- and item-based, plain or shrunk Pearson);
    kNN on the residuals of mean + biases (item-based, with / without our damping beta);
  * MF R ~ I U^T on raw ratings, after removing the mean, after removing mean + biases
    (src/mf_plain.py), and with biases learned jointly (src/mf.py); ALS and gradient
    descent; lambda ~ 0 (no regularization) and tuned.
"""
import csv
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run as a script from the repo root
from src.baselines import fit_biases, fit_global_mean
from src.data import evaluate, load_coo, split
from src.knn import fit_item_knn
from src.knn_course import fit_course_knn
from src.mf import fit_als, fit_sgd
from src.mf_plain import fit_plain_als, fit_plain_sgd

RANKS = (2, 5, 10, 20)
ALS_LAMS = (1.0, 5.0, 10.0, 20.0, 50.0)      # plain ALS: penalty summed over all ratings of a row
SGD_LAMS = (0.02, 0.05, 0.1)
NO_REG = 1e-3
KS = (10, 20, 40, 80)


def fit_one_side(train, shape, side, lam=2.0):
    """mean + regularized user (side=0) or movie (side=1) biases only."""
    keys, r = train[side], train[2]
    mu = r.mean()
    b = np.bincount(keys, r - mu, minlength=shape[side]) / (np.bincount(keys, minlength=shape[side]) + lam)
    return lambda u, i: mu + b[(u, i)[side]]


def variants():
    """{variant: [(settings dict, fit(train, shape)), ...]} of every setting tried."""
    v = {
        "global mean": [({}, fit_global_mean)],
        "mean + user biases": [(dict(lam=2.0), lambda d, s: fit_one_side(d, s, 0))],
        "mean + movie biases": [(dict(lam=2.0), lambda d, s: fit_one_side(d, s, 1))],
        "mean + both biases, no regularization": [(dict(lam=0.0), lambda d, s: fit_biases(d, s, 0.0))],
        "mean + both biases, regularized": [(dict(lam=l), lambda d, s, l=l: fit_biases(d, s, l)) for l in (0.5, 1, 2, 3, 5, 10)],
    }
    for axis, shrink in product(("user", "item"), (0.0, 100.0)):
        name = f"kNN course formula, {axis}-based, " + ("Pearson" if shrink == 0 else "shrunk Pearson")
        v[name] = [(dict(k=k), lambda d, s, k=k, a=axis, sh=shrink: fit_course_knn(d, s, k, sh, a)) for k in KS]
    for beta in (0.0, 0.2):
        name = "kNN on residuals of mean + biases, item-based" + (", damped" if beta else "")
        v[name] = [(dict(k=k), lambda d, s, k=k, b=beta: fit_item_knn(d, s, k, 100.0, b, 2.0)) for k in KS]
    for base in ("none", "mean", "biases"):
        v[f"MF on {base}, ALS, no regularization"] = [
            (dict(rank=k), lambda d, s, k=k, b=base: fit_plain_als(d, s, b, k, NO_REG)) for k in RANKS]
        v[f"MF on {base}, ALS, regularized"] = [
            (dict(rank=k, lam=l), lambda d, s, k=k, l=l, b=base: fit_plain_als(d, s, b, k, l)) for k, l in product(RANKS, ALS_LAMS)]
        v[f"MF on {base}, gradient descent, no regularization"] = [
            (dict(rank=k), lambda d, s, k=k, b=base: fit_plain_sgd(d, s, b, k, 0.0)) for k in RANKS]
        v[f"MF on {base}, gradient descent, regularized"] = [
            (dict(rank=k, lam=l), lambda d, s, k=k, l=l, b=base: fit_plain_sgd(d, s, b, k, l)) for k, l in product(RANKS, SGD_LAMS)]
    v["MF + biases learned jointly, ALS, regularized"] = [
        (dict(rank=k, lam_f=10.0, lam_b=2.0), lambda d, s, k=k: fit_als(d, s, k, 10.0, 2.0)) for k in RANKS]
    v["MF + biases learned jointly, gradient descent, regularized"] = [
        (dict(rank=k, lam_f=0.2, lam_b=0.05), lambda d, s, k=k: fit_sgd(d, s, k, 0.2, 0.05, 0.02, 17)) for k in RANKS]
    return v


def grid_rows(v, fit, valid, shape):
    """Validation RMSE of every setting of every variant."""
    return [dict(variant=name, settings=json.dumps(hp), valid_rmse=round(evaluate(f(fit, shape), valid)[0], 4))
            for name, settings in v.items() for hp, f in settings]


def best_rows(v, grid, train, test, shape):
    """Refit the best validation setting of each variant on all train, score train and test once."""
    rows = []
    for name, settings in v.items():
        scores = [g["valid_rmse"] for g in grid if g["variant"] == name]
        hp, f = settings[int(np.argmin(scores))]
        predict = f(train, shape)
        rows.append(dict(variant=name, settings=json.dumps(hp), train_rmse=round(evaluate(predict, train)[0], 4),
                         valid_rmse=min(scores), test_rmse=round(evaluate(predict, test)[0], 4)))
    return rows


def save_csv(path, rows):
    """Write a list of dicts as CSV."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    train, shape = load_coo("data/ratings_train.npy")
    test = load_coo("data/ratings_test.npy")[0]
    fit, valid = split(train, 0.1, 0)
    v = variants()
    grid = grid_rows(v, fit, valid, shape)
    save_csv("results/course_grid.csv", grid)
    best = best_rows(v, grid, train, test, shape)
    save_csv("results/course_results.csv", best)
    for row in best:
        print(row)
