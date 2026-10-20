"""Literature models not covered elsewhere: GC-MC (src/gcmc.py) and GLocal-K (src/glocalk.py).

Usage:
    python experiments/run_literature_models.py

Protocol (same as the other experiments): validation curves on the 10 % validation split
of the train ratings (seed 0) choose the number of epochs; the model fitted on the 90 %
part predicts the validation positions, the model refitted on the full train predicts
the test positions (scored once). Predictions are saved for blending under
results/preds/<name>_valid.npy and _test.npy (order of split(...)[1] and of
to_coo(ratings_test)). Writes results/lit_curves.csv and results/lit_results.csv.
Runs in ~15 min on 2 threads (GLocal-K dominates: 4-11 min per fit depending on CPU load).
"""
import os
os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "2"

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run as a script from the repo root
from src.data import clip, load_coo, rmse, split, to_coo
from src.gcmc import fit_gcmc, fit_gcmc_early
from src.glocalk import fit_glocalk, fit_glocalk_early
from src.side_info import item_features

ROOT = Path(__file__).resolve().parent.parent
RES, PREDS = ROOT / "results", ROOT / "results" / "preds"
F = item_features(ROOT / "data" / "namesngenre2.npy")
GCMC_HP = dict(epochs=200, dropout=0.7)
GLK_HP = dict(epochs_p=40, epochs_f=40)


def write_csv(rows, name):
    """Write a list of dicts to results/<name>."""
    with open(RES / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def save_preds(name, predict, data):
    """Save the clipped predictions at the positions of `data`; return their RMSE."""
    p = clip(predict(data[0], data[1]))
    np.save(PREDS / f"{name}.npy", p)
    return rmse(p, data[2])


def timed(fit):
    """(result of fit(), seconds)."""
    t = time.time()
    out = fit()
    return out, time.time() - t


def run_gcmc(fit, valid, train, test, shape):
    """GC-MC: curve, validation and test predictions; returns (curve rows, result row)."""
    _, curve = fit_gcmc_early(fit, shape, F, valid, epochs=300, dropout=GCMC_HP["dropout"])
    v = save_preds("gcmc_valid", fit_gcmc(fit, shape, F, **GCMC_HP), valid)
    predict, sec = timed(lambda: fit_gcmc(train, shape, F, **GCMC_HP))
    t = save_preds("gcmc_test", predict, test)
    rows = [dict(model="GC-MC", stage="train", epoch=e, valid_rmse=round(r, 4)) for e, r in curve]
    return rows, dict(model="GC-MC", hyperparams=json.dumps(GCMC_HP), valid_rmse=round(v, 4),
                      test_rmse=round(t, 4), fit_time_s=round(sec, 1))


def run_glocalk(fit, valid, train, test, shape):
    """GLocal-K: curve (its last epoch gives the validation predictions) and test predictions."""
    predict, curve = fit_glocalk_early(fit, shape, valid, **GLK_HP)
    v = save_preds("glocalk_valid", predict, valid)
    predict, sec = timed(lambda: fit_glocalk(train, shape, **GLK_HP))
    t = save_preds("glocalk_test", predict, test)
    rows = [dict(model="GLocal-K", stage=s, epoch=e, valid_rmse=round(r, 4)) for s, e, r in curve]
    return rows, dict(model="GLocal-K", hyperparams=json.dumps(GLK_HP), valid_rmse=round(v, 4),
                      test_rmse=round(t, 4), fit_time_s=round(sec, 1))


def main():
    """Run both models and write the CSV files."""
    torch.set_num_threads(2)
    PREDS.mkdir(exist_ok=True)
    train, shape = load_coo(ROOT / "data" / "ratings_train.npy")
    test = to_coo(np.load(ROOT / "data" / "ratings_test.npy"))
    fit, valid = split(train, 0.1, seed=0)
    curves, results = [], []
    for run in (run_gcmc, run_glocalk):
        c, r = run(fit, valid, train, test, shape)
        curves, results = curves + c, results + [r]
        print(r, flush=True)
        write_csv(curves, "lit_curves.csv")
        write_csv(results, "lit_results.csv")


if __name__ == "__main__":
    main()
