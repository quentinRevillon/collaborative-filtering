"""Models complementary to the Bayesian FM: EASE, kNN on residuals, GBDT residual correction.

Usage:
    python experiments/run_residual_models.py

Protocol: train ratings split into fit / valid (10 %, seed 0); every hyperparameter is
tuned on valid only, with every base model (biases, Bayesian FM, out-of-fold FMs)
fitted on the fit part. The selected configurations are refitted on the full train set
and scored once on the test set. The selected pipeline is finally refitted on
train+test and used to complete the whole matrix, timing the platform run.
Compared models:
  * EASE (src/ease.py), item-item and user-user, on bias and FM in-sample residuals;
  * item / user kNN (src/residual.py) on FM in-sample and out-of-fold residuals;
  * GBDT correction of the FM from out-of-fold residuals (src/residual_gbdt.py).
Writes results/residual_{grid,results,platform_time}.csv and, for the selected models,
results/preds/res_<model>_{valid,test}.npy (valid: fit on the fit part, at the valid
positions in split() order; test: fit on the full train, at to_coo(test) positions).
Runs in ~30 min on 3 CPU threads.
"""
import os
os.environ["TQDM_DISABLE"] = "1"   # silence myfm's progress bar
os.environ.setdefault("OMP_NUM_THREADS", "3")

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run as a script from the repo root
from src.baselines import fit_biases
from src.bfm import fit_bfm, fit_model, posterior_mean
from src.data import accuracy, clip, complete, load_coo, rmse, split
from src.ease import ease_correction, residual_matrix
from src.residual import fit_knn_correction, fold_ids, oof_predictions
from src.residual_gbdt import fit_gbdt_correction, fit_residual_gbdt
from src.side_info import item_features

ROOT = Path(__file__).resolve().parent.parent
RES, PREDS = ROOT / "results", ROOT / "results" / "preds"
FM_ITER, N_FOLDS = 2000, 5
FOLD_ITERS = [200, 500]                   # n_iter of the out-of-fold FMs
EASE_LAMS = [10.0, 100.0, 1000.0]
KNN_GRID = [(k, s, b) for k in (20, 80) for s in (30.0, 100.0) for b in (0.3, 1.0, 3.0)]
PRIORS = [3.0, 10.0, 30.0]                # pseudo-counts of the GBDT support statistics
F = item_features(ROOT / "data" / "namesngenre.npy")


def save_csv(rows, name):
    """Write a list of dicts as results/<name>."""
    with open(RES / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] {RES / name}")


def score(p, data):
    """Clipped RMSE of predictions p on the ratings of data."""
    return round(rmse(clip(p), data[2]), 4)


def row(model, params, valid_rmse):
    """One line of the validation grid."""
    print(f"{model:28s} {json.dumps(params):60s} {valid_rmse:.4f}", flush=True)
    return dict(model=model, hyperparams=json.dumps(params), valid_rmse=valid_rmse)


def fm_matrix(data, shape, n_iter):
    """Dense posterior-mean matrix of the Bayesian FM fitted on data."""
    return posterior_mean(*fit_model(data, shape, F, ("iu", "ii", "side"), 10, n_iter))


def lookup(M):
    """predict(u, i) closure reading a dense matrix."""
    return lambda u, i: M[u, i]


def fit_fold_fm(shape, n_iter):
    """fit_base(data) for the out-of-fold Bayesian FMs."""
    return lambda d: fit_bfm(d, shape, F, ("iu", "ii", "side"), 10, n_iter)


def ease_rows(fit, valid, shape, base, name):
    """EASE (item and user) on the in-sample residuals of base."""
    E, b = residual_matrix(fit, shape, base), base(valid[0], valid[1])
    return [row(f"EASE-{ax} on {name}", dict(lam=lam), score(b + ease_correction(E, lam, ax)[valid[0], valid[1]], valid))
            for ax in ("item", "user") for lam in EASE_LAMS]


def knn_rows(fit, valid, shape, e, b, name, extra):
    """Item/user kNN on the residuals e of the fit ratings, added to base predictions b."""
    return [row(f"kNN on {name}", dict(extra, axis=ax, k=k, shrink=s, beta=be),
                score(b + fit_knn_correction(fit, shape, e, k, s, be, ax)(valid[0], valid[1]), valid))
            for ax in ("item", "user") for k, s, be in KNN_GRID]


def gbdt_rows(fit, valid, shape, p_oof, fold, b, n_iter):
    """GBDT correction of the FM for every prior, given the out-of-fold FM predictions."""
    return [row("FM + GBDT(oof residuals)", dict(fold_iter=n_iter, prior=pr),
                score(b + fit_gbdt_correction(fit, shape, p_oof, fold, pr)(b, valid[0], valid[1]), valid))
            for pr in PRIORS]


def best(rows, prefix):
    """Best validation row among the models whose name starts with prefix."""
    return min((r for r in rows if r["model"].startswith(prefix)), key=lambda r: r["valid_rmse"])


def tune(fit, valid, shape):
    """Validation grid of every residual model; returns (rows, valid predictions of the base FM)."""
    rows = ease_rows(fit, valid, shape, fit_biases(fit, shape, 2.0), "biases")
    fm = lookup(fm_matrix(fit, shape, FM_ITER))
    b = fm(valid[0], valid[1])
    rows.append(row("Bayesian FM", dict(n_iter=FM_ITER), score(b, valid)))
    rows += ease_rows(fit, valid, shape, fm, "FM (in-sample)")
    rows += knn_rows(fit, valid, shape, fit[2] - fm(fit[0], fit[1]), b, "FM (in-sample)", {})
    fold = fold_ids(len(fit[2]), N_FOLDS)
    for n_iter in FOLD_ITERS:
        p_oof = oof_predictions(fit, fit_fold_fm(shape, n_iter), fold)
        rows += knn_rows(fit, valid, shape, fit[2] - p_oof, b, "FM (oof)", dict(fold_iter=n_iter))
        rows += gbdt_rows(fit, valid, shape, p_oof, fold, b, n_iter)
    return rows, b


def selected_predictions(data, shape, cfg_gbdt, cfg_knn, pos):
    """Predictions at pos of the base FM, FM + OOF-kNN and FM + GBDT, all fitted on data."""
    fm = lookup(fm_matrix(data, shape, FM_ITER))
    fold = fold_ids(len(data[2]), N_FOLDS)
    p_oof = {n: oof_predictions(data, fit_fold_fm(shape, n), fold) for n in {cfg_gbdt["fold_iter"], cfg_knn["fold_iter"]}}
    b = fm(*pos)
    knn = fit_knn_correction(data, shape, data[2] - p_oof[cfg_knn["fold_iter"]], cfg_knn["k"], cfg_knn["shrink"], cfg_knn["beta"], cfg_knn["axis"])
    gbdt = fit_gbdt_correction(data, shape, p_oof[cfg_gbdt["fold_iter"]], fold, cfg_gbdt["prior"])
    return {"bfm": b, "fm_oofknn": b + knn(*pos), "fm_gbdt": b + gbdt(b, *pos)}


def platform_time(train, test, shape, cfg):
    """Seconds to fit the selected FM + GBDT on train+test and complete the matrix."""
    data = tuple(np.concatenate(a) for a in zip(train, test))
    t = time.time()
    predict = fit_residual_gbdt(data, shape, fit_fold_fm(shape, FM_ITER), fit_fold_fm(shape, cfg["fold_iter"]),
                                N_FOLDS, cfg["prior"])
    complete(predict, shape)
    return time.time() - t


def main():
    """Tune on valid, score the selected models once on test, time the platform run."""
    train, shape = load_coo(ROOT / "data" / "ratings_train.npy")
    test, _ = load_coo(ROOT / "data" / "ratings_test.npy")
    fit, valid = split(train, 0.1, seed=0)
    rows, _ = tune(fit, valid, shape)
    save_csv(rows, "residual_grid.csv")

    cfg_gbdt = json.loads(best(rows, "FM + GBDT")["hyperparams"])
    cfg_knn = json.loads(best(rows, "kNN on FM (oof)")["hyperparams"])
    PREDS.mkdir(parents=True, exist_ok=True)
    valid_p = selected_predictions(fit, shape, cfg_gbdt, cfg_knn, valid[:2])
    test_p = selected_predictions(train, shape, cfg_gbdt, cfg_knn, test[:2])
    out = []
    for name, cfg in [("bfm", dict(n_iter=FM_ITER)), ("fm_oofknn", cfg_knn), ("fm_gbdt", cfg_gbdt)]:
        np.save(PREDS / f"res_{name}_valid.npy", valid_p[name])
        np.save(PREDS / f"res_{name}_test.npy", test_p[name])
        out.append(dict(model=name, hyperparams=json.dumps(cfg), valid_rmse=score(valid_p[name], valid),
                        test_rmse=score(test_p[name], test), test_acc=round(accuracy(clip(test_p[name]), test[2]), 4)))
        print(out[-1], flush=True)
    save_csv(out, "residual_results.csv")
    sec = platform_time(train, test, shape, cfg_gbdt)
    save_csv([dict(model="fm_gbdt", hyperparams=json.dumps(cfg_gbdt), n_ratings=len(train[2]) + len(test[2]),
                   threads=os.environ["OMP_NUM_THREADS"], fit_complete_time_s=round(sec, 1))], "residual_platform_time.csv")


if __name__ == "__main__":
    main()
