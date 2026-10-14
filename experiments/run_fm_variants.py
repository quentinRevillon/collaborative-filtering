"""Variants of the Bayesian FM (src/bfm.py): ordered probit likelihood, extra features, budget, seeds.

Usage:
    python experiments/run_fm_variants.py valid     # grid on the validation split
    python experiments/run_fm_variants.py test      # selected config: test score, preds, timing

Protocol: train = data/ratings_train.npy; (fit, valid) = split(train, 0.1, seed=0). Every
configuration of GRID is fitted on `fit` and scored on `valid` (clipped predictions);
SELECTED is chosen on valid only. The "test" stage refits SELECTED on `fit` (valid
predictions saved for blending), then on the full train set and scores it once on
data/ratings_test.npy; finally it times the platform run: fit on train+test merged and
complete the whole 610 x 4980 matrix.
Writes results/fm_variants_valid.csv, results/fm_variants_test.csv and
results/preds/fm_variants_{valid,test}.npy (order of split()'s valid triplets / to_coo(test)).
Configs run in parallel in WORKERS single-threaded processes.
"""
import os
os.environ["TQDM_DISABLE"] = "1"  # silence myfm's progress bar
os.environ["OMP_NUM_THREADS"] = "1"

import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run as a script from the repo root
from src.bfm import fit_model, posterior_mean
from src.bfm_ordinal import expected_dense, fit_ordinal_model
from src.data import accuracy, clip, complete, load_coo, rmse, split
from src.side_info import item_features

ROOT = Path(__file__).resolve().parent.parent
RES, PREDS = ROOT / "results", ROOT / "results" / "preds"
F = item_features(ROOT / "data" / "namesngenre.npy")
WORKERS = 3
BASE = ("iu", "ii", "side")
# (likelihood, features, rank, n_iter, seeds); round 1 "us"/"ip" also had a shrunk mean rating (leaky, 0.905)
GRID = [
    ("reg", BASE, 10, 1000, (0,)),
    ("ord", BASE, 10, 1000, (0,)),
    ("ord", BASE + ("iur",), 10, 1000, (0,)),
    ("ord", BASE + ("iir",), 10, 1000, (0,)),
    ("ord", BASE + ("iur", "iir"), 10, 1000, (0,)),
    ("ord", BASE + ("us", "ip"), 10, 1000, (0,)),
    ("reg", BASE + ("iur", "iir"), 10, 1000, (0,)),
    ("reg", BASE + ("us", "ip"), 10, 1000, (0,)),
]
GRID2 = [
    ("reg", BASE + ("iur",), 10, 1000, (0,)),
    ("reg", BASE + ("us", "ip"), 10, 1000, (0,)),
    ("ord", BASE, 10, 2000, (0,)),
    ("reg", BASE, 8, 1000, (0,)),
    ("reg", BASE, 12, 1000, (0,)),
    ("reg", BASE, 16, 1000, (0,)),
    ("reg", BASE, 10, 3000, (0,)),
    ("reg", BASE, 10, 1000, (0, 1, 2)),
]
GRID3 = [
    ("mix", BASE, 10, 1000, (0,)),
    ("reg", BASE, 10, 1000, (0, 1, 2, 3, 4, 5)),
    ("reg", BASE, 10, 2000, (0, 1, 2)),
    ("reg", BASE, 16, 2000, (0,)),
]
SELECTED = ("reg", BASE, 10, 2000, (0, 1, 2))     # best valid RMSE (GRID3)


def dense_chain(kind, train, shape, features, rank, n_iter, seed):
    """Dense users x items prediction of one Gibbs chain: "reg" posterior mean, "ord" expected
    rating, "mix" the average of both."""
    if kind == "reg":
        return posterior_mean(*fit_model(train, shape, F, features, rank, n_iter, seed))
    if kind == "ord":
        return expected_dense(*fit_ordinal_model(train, shape, F, features, rank, n_iter, seed))
    return (dense_chain("reg", train, shape, features, rank, n_iter, seed)
            + dense_chain("ord", train, shape, features, rank, n_iter, seed)) / 2


def fit_dense(config, train, shape):
    """Fit `config` on train and return the dense prediction averaged over its seeds (chains)."""
    kind, features, rank, n_iter, seeds = config
    return np.mean([dense_chain(kind, train, shape, features, rank, n_iter, s) for s in seeds], 0)


def fit_predict(config, train, shape, u, i):
    """Fit `config` on train and return its predictions at the positions (u, i)."""
    return fit_dense(config, train, shape)[u, i]


def describe(config):
    """Readable row fields of a config."""
    kind, features, rank, n_iter, seeds = config
    return {"likelihood": kind, "features": "+".join(features), "rank": rank, "n_iter": n_iter,
            "seeds": len(seeds)}


def valid_row(config):
    """Fit config on the fit split and score it on valid (one CSV row)."""
    train, shape = load_coo(ROOT / "data" / "ratings_train.npy")
    fit, valid = split(train, 0.1, seed=0)
    t = time.time()
    p = clip(fit_predict(config, fit, shape, valid[0], valid[1]))
    row = {**describe(config), "valid_rmse": round(rmse(p, valid[2]), 4),
           "valid_acc": round(accuracy(p, valid[2]), 4), "time_s": round(time.time() - t, 1)}
    print(row, flush=True)
    return row


def save_csv(rows, name):
    """Write a list of dicts as results/<name>."""
    with open(RES / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] {RES / name}")


def run_valid(grid, name):
    """Score every config of grid on the validation split; write results/<name>."""
    with Pool(WORKERS) as pool:
        save_csv(pool.map(valid_row, grid, chunksize=1), name)


def merged(a, b):
    """Concatenate two COO triplets."""
    return tuple(np.concatenate([x, y]) for x, y in zip(a, b))


def run_test():
    """SELECTED: valid preds (fit split), test score (full train), platform timing (train+test)."""
    train, shape = load_coo(ROOT / "data" / "ratings_train.npy")
    test, _ = load_coo(ROOT / "data" / "ratings_test.npy")
    fit, valid = split(train, 0.1, seed=0)
    PREDS.mkdir(exist_ok=True)
    pv = fit_predict(SELECTED, fit, shape, valid[0], valid[1])
    np.save(PREDS / "fm_variants_valid.npy", pv)
    pt = fit_predict(SELECTED, train, shape, test[0], test[1])
    np.save(PREDS / "fm_variants_test.npy", pt)
    t = time.time()
    M = fit_dense(SELECTED, merged(train, test), shape)
    full = complete(lambda u, i: M[u, i], shape)
    row = {**describe(SELECTED), "valid_rmse": round(rmse(clip(pv), valid[2]), 4),
           "test_rmse": round(rmse(clip(pt), test[2]), 4), "test_acc": round(accuracy(clip(pt), test[2]), 4),
           "train_test_time_s": round(time.time() - t, 1), "full_min": float(full.min())}
    print(row)
    save_csv([row], "fm_variants_test.csv")


if __name__ == "__main__":
    {"valid": lambda: run_valid(GRID, "fm_variants_valid.csv"),
     "valid2": lambda: run_valid(GRID2, "fm_variants_valid2.csv"),
     "valid3": lambda: run_valid(GRID3, "fm_variants_valid3.csv"),
     "test": run_test}[sys.argv[1]]()
