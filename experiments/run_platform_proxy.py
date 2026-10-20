"""Local proxy of the course platform: fit on train + half of test, score the other half.

Usage:
    python experiments/run_platform_proxy.py                 # all models
    python experiments/run_platform_proxy.py biases knn      # a subset (names of MODELS)

Protocol. The platform fits on train + test (63,196 ratings) and scores a hidden set.
Locally, the test ratings are randomly split in two halves A / B (seed 0). Fold 1 fits
on train + A and scores B, fold 2 fits on train + B and scores A; the proxy RMSE is the
mean of the two folds. Every model is also fitted on two smaller inputs, scored on the
same halves: train only (31,598 ratings) and a 75 % random subsample of train (seed 0),
which gives the learning curve RMSE(n) at n = 23.7k, 31.6k, 47.4k. A fit
RMSE = a + b * n^(-1/2) through these points is extrapolated to the platform size n = 63,196.
Hyperparameters are fixed beforehand (selected on the 10 % validation split of train in
the other experiments): nothing is tuned on the test halves.
Writes results/proxy_rmse.csv (one row per model x input x fold) and
results/proxy_summary.csv (per model: RMSE per input size, gain, extrapolation).
"""
import os
os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "2"
os.environ["TQDM_DISABLE"] = "1"

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run as a script from the repo root
from src.baselines import fit_biases, fit_global_mean
from src.bfm import fit_bfm
from src.data import evaluate, load_coo, split, to_coo
from src.gcmc import fit_gcmc
from src.knn import fit_item_knn
from src.side_info import item_features

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
F = item_features(ROOT / "data" / "namesngenre2.npy")
PLATFORM_N = 63196

MODELS = {
    "global mean": lambda d, s: fit_global_mean(d, s),
    "biases": lambda d, s: fit_biases(d, s, lam=2.0),
    "item-kNN": lambda d, s: fit_item_knn(d, s, k=40, shrink=100.0, beta=0.2, lam=2.0),
    "Bayesian FM": lambda d, s: fit_bfm(d, s, F, features=("iu", "ii", "side"), rank=10, n_iter=2000),
    "GC-MC": lambda d, s: fit_gcmc(d, s, F, epochs=200, dropout=0.7),
}


def concat(a, b):
    """Concatenate two COO triplets."""
    return tuple(np.concatenate([x, y]) for x, y in zip(a, b))


def test_halves(test, seed=0):
    """Random split of the test ratings into two halves (A, B)."""
    return split(test, 0.5, seed)


def inputs(train, half):
    """Model inputs of one fold: {name: COO} for 75 % of train, train, train + half."""
    return {"train75": split(train, 0.25, seed=0)[0], "train": train, "train+half": concat(train, half)}


def run_fold(name, train, shape, fit_half, score_half, fold):
    """Fit `name` on every input of the fold and score `score_half`; list of result rows."""
    rows = []
    for key, data in inputs(train, fit_half).items():
        t = time.time()
        predict = MODELS[name](data, shape)
        rows.append(dict(model=name, input=key, n_fit=len(data[2]), fold=fold,
                         rmse=round(evaluate(predict, score_half)[0], 4), time_s=round(time.time() - t, 1)))
        print(rows[-1], flush=True)
    return rows


def extrapolate(ns, rmses, n):
    """Least-squares fit of RMSE = a + b / sqrt(n) on the points, evaluated at n."""
    b, a = np.polyfit(1 / np.sqrt(ns), rmses, 1)
    return a + b / np.sqrt(n)


def summarize(rows):
    """Per-model summary: mean RMSE over folds per input, gain train -> train+half, extrapolation."""
    out = []
    for name in dict.fromkeys(r["model"] for r in rows):
        m = {k: np.mean([r["rmse"] for r in rows if r["model"] == name and r["input"] == k])
             for k in ("train75", "train", "train+half")}
        n = {k: np.mean([r["n_fit"] for r in rows if r["input"] == k]) for k in m}
        out.append(dict(model=name, rmse_train75=round(m["train75"], 4), rmse_train=round(m["train"], 4),
                        rmse_proxy=round(m["train+half"], 4), gain_train_to_proxy=round(m["train"] - m["train+half"], 4),
                        est_platform=round(extrapolate(list(n.values()), list(m.values()), PLATFORM_N), 4)))
    return out


def write_csv(rows, name):
    """Write a list of dicts to results/<name>."""
    with open(RES / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(names):
    """Run the proxy protocol for the given model names and write the CSV files."""
    torch.set_num_threads(2)
    train, shape = load_coo(ROOT / "data" / "ratings_train.npy")
    test = to_coo(np.load(ROOT / "data" / "ratings_test.npy"))
    A, B = test_halves(test)
    rows = [r for name in names for r in run_fold(name, train, shape, A, B, 1) + run_fold(name, train, shape, B, A, 2)]
    tag = "" if names == list(MODELS) else "_" + "_".join(n.replace(" ", "").replace("-", "") for n in names)
    write_csv(rows, f"proxy_rmse{tag}.csv")
    write_csv(summarize(rows), f"proxy_summary{tag}.csv")
    print(summarize(rows))


if __name__ == "__main__":
    main(sys.argv[1:] or list(MODELS))
