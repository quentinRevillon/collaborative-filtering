"""Matrix factorization experiments: biased ALS, ALS-WR and SGD (Funk-SVD).

Protocol: hyperparameters are chosen on a 10 % validation split of ratings_train
(seed 0); the best configuration of each method is refit on the full train set and
scored once on ratings_test (never used for tuning).

Usage:
    python experiments/run_mf.py

Writes results/mf_grid.csv (validation grid), results/mf_final.csv (test scores)
and figures/mf_*.png. Runtime ~10 min on a laptop CPU.
"""
import csv
import itertools
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baselines import fit_biases  # noqa: E402
from src.data import evaluate, load_coo, split  # noqa: E402
from src.mf import als_steps, fit_als, fit_sgd, history, sgd_steps  # noqa: E402

RESULTS, FIGURES = Path("results"), Path("figures")
RANKS = [1, 2, 5, 10, 20, 50]
ALS_ITER = 15
ALS_GRID = {  # wr -> lam_f grid (log scale); the WR penalty is lam_f * n_ratings
    False: [3.0, 5.0, 10.0, 15.0, 30.0],
    True: [0.1, 0.2, 0.3, 0.5, 1.0],
}
ALS_LAM_B = [1.0, 2.0, 5.0]
SGD_GRID = dict(rank=[5, 20], lr=[0.005, 0.01, 0.02], lam_f=[0.05, 0.1, 0.2, 0.4], lam_b=[0.05, 0.3])
SGD_EPOCHS = 60
BIAS_LAM = 2.0
COLORS = {"biases": "#52514e", "als": "#2a78d6", "als_wr": "#eb6834", "sgd": "#1baf7a"}
FIELDS = ["method", "rank", "lam_f", "lam_b", "lr", "n_iter", "valid_rmse", "test_rmse",
          "test_acc", "fit_time"]


def timed_fit(fit, *args, **kw):
    """Return (predict, seconds) for fit(*args, **kw)."""
    t = time.perf_counter()
    predict = fit(*args, **kw)
    return predict, time.perf_counter() - t


def als_row(fit_set, valid, shape, rank, lam_f, lam_b, wr):
    """Fit one ALS configuration on fit_set and return its grid row."""
    predict, sec = timed_fit(fit_als, fit_set, shape, rank, lam_f, lam_b, ALS_ITER, wr)
    return dict(method="als_wr" if wr else "als", rank=rank, lam_f=lam_f, lam_b=lam_b,
                n_iter=ALS_ITER, valid_rmse=evaluate(predict, valid)[0], fit_time=sec)


def sgd_row(fit_set, valid, shape, rank, lr, lam_f, lam_b):
    """Run one SGD configuration, keep the best epoch on valid (early stopping) as its row."""
    t = time.perf_counter()
    curve = [v for _, v in history(sgd_steps(fit_set, shape, rank, lam_f, lam_b, lr, SGD_EPOCHS),
                                   fit_set, valid)]
    best = int(np.nanargmin(curve))
    sec = (time.perf_counter() - t) * (best + 1) / SGD_EPOCHS
    return dict(method="sgd", rank=rank, lam_f=lam_f, lam_b=lam_b, lr=lr, n_iter=best + 1,
                valid_rmse=curve[best], fit_time=sec)


def als_grid(fit_set, valid, shape):
    """Evaluate every (wr, rank, lam_f, lam_b) ALS configuration on valid."""
    return [als_row(fit_set, valid, shape, k, lf, lb, wr)
            for wr, lams in ALS_GRID.items()
            for k, lf, lb in itertools.product(RANKS, lams, ALS_LAM_B)]


def sgd_grid(fit_set, valid, shape):
    """Evaluate every SGD configuration of SGD_GRID on valid."""
    return [sgd_row(fit_set, valid, shape, *cfg) for cfg in itertools.product(*SGD_GRID.values())]


def best(rows, method):
    """Return the row of `method` with the lowest validation RMSE."""
    return min((r for r in rows if r["method"] == method), key=lambda r: r["valid_rmse"])


def refit(row, train, test, shape):
    """Refit a grid row on the full train set and add its test scores and fit time."""
    if row["method"] == "sgd":
        predict, sec = timed_fit(fit_sgd, train, shape, row["rank"], row["lam_f"], row["lam_b"],
                                 row["lr"], row["n_iter"])
    else:
        predict, sec = timed_fit(fit_als, train, shape, row["rank"], row["lam_f"], row["lam_b"],
                                 row["n_iter"], row["method"] == "als_wr")
    rmse, acc = evaluate(predict, test)
    return {**row, "test_rmse": rmse, "test_acc": acc, "fit_time": sec}


def biases_row(fit_set, valid, train, test, shape):
    """Score the biases baseline (lam=BIAS_LAM) on valid and, refit on train, on test."""
    predict, sec = timed_fit(fit_biases, train, shape, BIAS_LAM)
    rmse, acc = evaluate(predict, test)
    return dict(method="biases", lam_b=BIAS_LAM, valid_rmse=evaluate(fit_biases(fit_set, shape, BIAS_LAM),
                valid)[0], test_rmse=rmse, test_acc=acc, fit_time=sec)


def save_csv(rows, path):
    """Write rows (dicts) to a CSV file with the FIELDS columns."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def plot_learning_curves(fit_set, valid, shape, als_best, sgd_best, path):
    """Plot train vs valid RMSE per iteration/epoch, unregularized vs tuned, for ALS and SGD."""
    k = 50
    runs = {
        "ALS": [("lam ~ 0", als_steps(fit_set, shape, k, 1e-3, 1e-3, ALS_ITER)),
                (f"lam_f={als_best['lam_f']:g}, lam_b={als_best['lam_b']:g}",
                 als_steps(fit_set, shape, k, als_best["lam_f"], als_best["lam_b"], ALS_ITER))],
        "SGD": [("lam = 0", sgd_steps(fit_set, shape, k, 0.0, 0.0, sgd_best["lr"], SGD_EPOCHS)),
                (f"lam_f={sgd_best['lam_f']:g}, lam_b={sgd_best['lam_b']:g}",
                 sgd_steps(fit_set, shape, k, sgd_best["lam_f"], sgd_best["lam_b"], sgd_best["lr"],
                           SGD_EPOCHS))],
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (name, pair) in zip(axes, runs.items()):
        for (label, steps), color in zip(pair, [COLORS["als_wr"], COLORS["als"]]):
            h = np.array(history(steps, fit_set, valid))
            x = np.arange(1, len(h) + 1)
            ax.plot(x, h[:, 0], "--", color=color, lw=2, label=f"train, {label}")
            ax.plot(x, h[:, 1], "-", color=color, lw=2, label=f"valid, {label}")
        ax.set(title=f"{name}, rank {k}", xlabel="iteration" if name == "ALS" else "epoch",
               ylabel="RMSE")
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=9)
    finish(fig, path)


def plot_rank(rows, bias_valid, path):
    """Plot valid RMSE vs rank for each lam_f (lam_b at its best value), ALS and ALS-WR."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, wr in zip(axes, [False, True]):
        method = "als_wr" if wr else "als"
        lam_b = best(rows, method)["lam_b"]
        lams = ALS_GRID[wr]
        for lf, color in zip(lams, plt.cm.Blues(np.linspace(0.35, 1, len(lams)))):
            pts = sorted((r["rank"], r["valid_rmse"]) for r in rows
                         if r["method"] == method and r["lam_f"] == lf and r["lam_b"] == lam_b)
            ax.plot(*zip(*pts), "o-", color=color, lw=2, ms=6, label=f"lam_f = {lf:g}")
        ax.axhline(bias_valid, color=COLORS["biases"], ls=":", lw=1.5, label="biases only")
        ax.set(xscale="log", xticks=RANKS, xticklabels=RANKS, xlabel="rank k",
               title=f"{'ALS-WR' if wr else 'ALS'} (lam_b = {lam_b:g})", ylim=(0.88, 0.96))
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("validation RMSE")
    finish(fig, path)


def plot_time(rows, path):
    """Scatter validation RMSE against fit time for every grid configuration."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for method, label in [("als", "ALS"), ("als_wr", "ALS-WR"), ("sgd", "SGD (best epoch)")]:
        pts = [(r["fit_time"], r["valid_rmse"]) for r in rows if r["method"] == method]
        ax.scatter(*zip(*pts), s=28, color=COLORS[method], alpha=0.75, edgecolor="white",
                   label=label)
    ax.set(xscale="log", xticks=[0.3, 1, 3], xticklabels=["0.3", "1", "3"],
           xlabel="fit time on 90 % of train (s, log scale)",
           ylabel="validation RMSE", ylim=(0.88, 1.0))
    ax.legend(frameon=False)
    finish(fig, path)


def finish(fig, path):
    """Apply a clean style to all axes, save the figure and close it."""
    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    """Run the grids, the refits on train, the test scoring, and save CSVs and figures."""
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    train, shape = load_coo("data/ratings_train.npy")
    test, _ = load_coo("data/ratings_test.npy")
    fit_set, valid = split(train, frac=0.1, seed=0)
    rows = als_grid(fit_set, valid, shape) + sgd_grid(fit_set, valid, shape)
    save_csv(rows, RESULTS / "mf_grid.csv")
    final = [biases_row(fit_set, valid, train, test, shape)]
    final += [refit(best(rows, m), train, test, shape) for m in ["als", "als_wr", "sgd"]]
    save_csv(final, RESULTS / "mf_final.csv")
    for r in final:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    plot_learning_curves(fit_set, valid, shape, best(rows, "als"), best(rows, "sgd"),
                         FIGURES / "mf_learning_curves.png")
    plot_rank(rows, final[0]["valid_rmse"], FIGURES / "mf_rank.png")
    plot_time(rows, FIGURES / "mf_time.png")


if __name__ == "__main__":
    main()
