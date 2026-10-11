"""Bayesian models by MCMC: BPMF with biases (src/bpmf.py) and Bayesian FM (src/bfm.py, myfm).

Usage:
    python experiments/run_bayesian.py

Protocol: hyperparameters are tuned on a 10 % validation split of the train ratings
(seed 0); the selected configurations are refitted on the full train set and scored
once on the test set. The Bayesian FM is also refitted on train+test and used to
complete the whole matrix, timing the run the platform would do (4 threads).
BPMF is only timed on train (it is not the recommended model).
Writes results/bayes_*.csv and figures/bayes/*.png. Runs in ~20 min on 4 CPU threads.
"""
import os
os.environ["TQDM_DISABLE"] = "1"  # silence myfm's progress bar (read when tqdm is imported)

import csv
import json
import sys
import time
from itertools import islice
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run as a script from the repo root
from src.bfm import BURN_IN, fit_bfm, fit_model, sample_scores
from src.bpmf import fit_bpmf, gibbs_samples
from src.data import clip, complete, evaluate, load_coo, rmse, split
from src.side_info import item_features

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figures" / "bayes"
THREADS = 4
BPMF_SWEEPS, BPMF_BURN = 400, 100            # tuning runs: 300 averaged samples
BPMF_ALPHAS = [None, 1.5, 2.0]               # None = noise precision sampled
BPMF_RANKS = [5, 10, 20]
BFM_ITER, BFM_CURVE_ITER = 1000, 2000
BFM_RANKS = [5, 10, 20]
ABLATION = {"MF (u, i)": (), "+ implicite user (SVD++)": ("iu",), "+ implicite item (flipped)": ("ii",),
            "+ genres/décennie": ("side",), "tout": ("iu", "ii", "side")}
REFS = {"MF-ALS": "#1baf7a", "item-kNN": "#eb6834", "blend (ridge)": "#eda100"}  # reference -> color
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#52514e"]
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3})


def save_csv(rows, name):
    """Write a list of dicts as results/<name>."""
    with open(RES / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] {RES / name}")


def save_fig(fig, name):
    """Save a figure in figures/bayes/ and close it."""
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {FIG / name}")


def reference_rmse():
    """Valid / test RMSE of the reference models of results/alt_results.csv."""
    with open(RES / "alt_results.csv") as f:
        return {r["method"]: (float(r["valid_rmse"]), float(r["test_rmse"])) for r in csv.DictReader(f)
                if r["method"] in REFS}


def mcmc_curve(single, valid):
    """Valid RMSE per sample index of single draws and of their running mean.

    single: (n_samples, n_valid) per-sample predictions at the valid positions.
    """
    mean = np.cumsum(single, 0) / np.arange(1, len(single) + 1)[:, None]
    return ([rmse(clip(p), valid[2]) for p in single], [rmse(clip(p), valid[2]) for p in mean])


def bpmf_history(tr, va, shape, hp):
    """Per-sample valid predictions of one BPMF run (after burn-in) and its run time."""
    t = time.perf_counter()
    samples = islice(gibbs_samples(tr, shape, hp["rank"], hp["alpha"]), BPMF_BURN, BPMF_SWEEPS)
    single = np.array([M[va[0], va[1]] for M in samples])
    return single, time.perf_counter() - t


def tune_bpmf(tr, va, shape):
    """Alpha grid at rank 10, then rank grid at the best alpha; returns (rows, curves, best)."""
    rows, curves = [], {}
    for hp in [dict(rank=10, alpha=a) for a in BPMF_ALPHAS]:
        curves[json.dumps(hp)] = bpmf_row(tr, va, shape, hp, rows)
    best = min(rows, key=lambda r: r["valid_rmse"])
    for hp in [dict(rank=k, alpha=best["alpha"]) for k in BPMF_RANKS if k != 10]:
        curves[json.dumps(hp)] = bpmf_row(tr, va, shape, hp, rows)
    best = min(rows, key=lambda r: r["valid_rmse"])
    return rows, curves, dict(rank=best["rank"], alpha=best["alpha"])


def bpmf_row(tr, va, shape, hp, rows):
    """Run one BPMF configuration, append its grid row to rows and return its MCMC curve."""
    single, sec = bpmf_history(tr, va, shape, hp)
    curve = mcmc_curve(single, va)
    rows.append(dict(hp, n_samples=BPMF_SWEEPS - BPMF_BURN, burn_in=BPMF_BURN,
                     valid_rmse=round(curve[1][-1], 4), time_s=round(sec, 1)))
    print(f"  BPMF {rows[-1]}", flush=True)
    return curve


def bfm_row(tr, va, shape, F, name, features, rank, n_iter):
    """Fit one Bayesian FM on tr; return (grid row, MCMC curve on va).

    valid_rmse uses the first BFM_ITER iterations (comparable rows), valid_rmse_last all n_iter.
    """
    t = time.perf_counter()
    model, blocks = fit_model(tr, shape, F, features, rank, n_iter)
    curve = mcmc_curve(sample_scores(model, blocks, va[0], va[1]), va)
    row = dict(config=name, features="+".join(features) or "none", rank=rank, n_iter=n_iter,
               valid_rmse=round(curve[1][BFM_ITER - BURN_IN - 1], 4), valid_rmse_last=round(curve[1][-1], 4),
               time_s=round(time.perf_counter() - t, 1))
    print(f"  BFM {row}", flush=True)
    return row, curve


def tune_bfm(tr, va, shape, F):
    """Long run (MCMC curve), feature ablation and rank grid at BFM_ITER iterations.

    Returns (rows, curve of the long rank-10 run, best rank of the all-features model).
    """
    rows = []
    row, curve = bfm_row(tr, va, shape, F, "tout", ABLATION["tout"], 10, BFM_CURVE_ITER)
    for name, feats in ABLATION.items():
        rows.append(row if name == "tout" else bfm_row(tr, va, shape, F, name, feats, 10, BFM_ITER)[0])
    for k in BFM_RANKS:
        if k != 10:
            rows.append(bfm_row(tr, va, shape, F, f"tout, rang {k}", ABLATION["tout"], k, BFM_ITER)[0])
    best = min((r for r in rows if r["features"] == "iu+ii+side"), key=lambda r: r["valid_rmse"])
    return rows, curve, best["rank"]


def test_row(name, fit, hp, valid_rmse, train, test, shape):
    """Refit on the full train set, score once on test; returns the result row."""
    t = time.perf_counter()
    predict = fit(train, shape, **hp)
    sec = time.perf_counter() - t
    r, a = evaluate(predict, test)
    hp = {k: v for k, v in hp.items() if k != "F"}
    print(f"{name:12s} valid={valid_rmse:.4f} test={r:.4f} acc={a:.4f} fit={sec:.0f}s {hp}", flush=True)
    return dict(method=name, hyperparams=json.dumps(hp), valid_rmse=valid_rmse, test_rmse=round(r, 4),
                test_acc=round(a, 4), fit_time_s=round(sec, 1))


def platform_time(fit, hp, data, shape):
    """Seconds to fit on `data` (train+test) and complete the whole matrix, as generate.py would."""
    t = time.perf_counter()
    complete(fit(data, shape, **hp), shape)
    return time.perf_counter() - t


def curve_rows(model, curve):
    """Long-format rows (model, n_samples, single-sample and running-mean valid RMSE) of a curve."""
    return [dict(model=model, n_samples=k + 1, single_rmse=round(s, 4), mean_rmse=round(m, 4))
            for k, (s, m) in enumerate(zip(*curve))]


def plot_samples(bpmf_curve, bfm_curve, refs):
    """Valid RMSE vs number of MCMC samples: single draws vs running posterior mean."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, (name, (single, mean)) in zip(axes, [("BPMF", bpmf_curve), ("Bayesian FM (toutes features)", bfm_curve)]):
        x = np.arange(1, len(mean) + 1)
        ax.plot(x, single, color=COLORS[5], lw=0.6, alpha=0.6, label="un seul échantillon")
        ax.plot(x, mean, color=COLORS[0], lw=2.2, label="moyenne des échantillons")
        for ref, (v, _) in refs.items():
            ax.axhline(v, color=REFS[ref], ls="--", lw=1.3, label=ref)
        ax.set(xscale="log", xlabel="nombre d'échantillons moyennés (après burn-in)", title=name)
    axes[0].set(ylabel="RMSE validation", ylim=(0.86, 1.08))
    axes[1].legend(frameon=False, fontsize=9)
    save_fig(fig, "rmse_vs_samples.png")


def plot_rank(bpmf_rows, bfm_rows, refs):
    """Valid RMSE vs rank for BPMF (best alpha) and the Bayesian FM with all features."""
    fig, ax = plt.subplots(figsize=(7, 4.4))
    alpha = min(bpmf_rows, key=lambda r: r["valid_rmse"])["alpha"]
    pts = sorted((r["rank"], r["valid_rmse"]) for r in bpmf_rows if r["alpha"] == alpha)
    ax.plot(*zip(*pts), marker="o", color=COLORS[0], lw=2, label=f"BPMF (α = {alpha or 'échantillonné'})")
    pts = sorted((r["rank"], r["valid_rmse"]) for r in bfm_rows if r["features"] == "iu+ii+side")
    ax.plot(*zip(*pts), marker="s", color=COLORS[4], lw=2, label="Bayesian FM (toutes features)")
    for ref, (v, _) in refs.items():
        ax.axhline(v, color=REFS[ref], ls="--", lw=1.3, label=ref)
    ax.set(xlabel="rang", ylabel="RMSE validation", xticks=[p[0] for p in pts],
           title=f"RMSE validation selon le rang ({BFM_ITER} itérations / {BPMF_SWEEPS} sweeps)")
    ax.legend(frameon=False, fontsize=9)
    save_fig(fig, "rmse_vs_rank.png")


def plot_ablation(bfm_rows, refs):
    """Horizontal bars of the valid RMSE of each feature group of the Bayesian FM (rank 10)."""
    rows = [r for r in bfm_rows if r["config"] in ABLATION]
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(rows) + 1.8))
    vals = [r["valid_rmse"] for r in rows]
    ax.barh([r["config"] for r in rows], vals, color=COLORS[0], height=0.6)
    for y, v in enumerate(vals):
        ax.text(v, y, f" {v:.4f}", va="center", fontsize=10)
    for ref, (v, _) in refs.items():
        ax.axvline(v, color=REFS[ref], ls="--", lw=1.3, label=ref)
    ax.set(xlim=(min(vals) - 0.01, max(vals + [v for v, _ in refs.values()]) + 0.006),
           xlabel="RMSE validation", title=f"Bayesian FM : ablation des features (rang 10, {BFM_ITER} itérations)")
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    save_fig(fig, "ablation.png")


def main():
    """Tune, refit and score BPMF and the Bayesian FM; write CSVs and figures."""
    torch.set_num_threads(THREADS)
    RES.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    train, shape = load_coo(ROOT / "data" / "ratings_train.npy")
    test, _ = load_coo(ROOT / "data" / "ratings_test.npy")
    F = item_features(ROOT / "data" / "namesngenre.npy")
    tr, va = split(train, 0.1, seed=0)
    refs = reference_rmse()

    bpmf_rows, bpmf_curves, bpmf_hp = tune_bpmf(tr, va, shape)
    save_csv(bpmf_rows, "bayes_bpmf_grid.csv")
    bfm_rows, bfm_curve, bfm_rank = tune_bfm(tr, va, shape, F)
    save_csv(bfm_rows, "bayes_bfm_grid.csv")
    save_csv(curve_rows("BPMF", bpmf_curves[json.dumps(bpmf_hp)]) + curve_rows("Bayesian FM", bfm_curve),
             "bayes_samples.csv")
    plot_samples(bpmf_curves[json.dumps(bpmf_hp)], bfm_curve, refs)
    plot_rank(bpmf_rows, bfm_rows, refs)
    plot_ablation(bfm_rows, refs)

    bpmf_full = dict(bpmf_hp, n_samples=BPMF_SWEEPS - BPMF_BURN, burn_in=BPMF_BURN)
    bfm_full = dict(F=F, features=ABLATION["tout"], rank=bfm_rank, n_iter=BFM_CURVE_ITER)
    bfm_valid = round(bfm_curve[1][-1], 4) if bfm_rank == 10 else min(
        r["valid_rmse"] for r in bfm_rows if r["features"] == "iu+ii+side")  # 2000 vs 1000 iterations
    results = [test_row("BPMF", fit_bpmf, bpmf_full, min(r["valid_rmse"] for r in bpmf_rows), train, test, shape),
               test_row("Bayesian FM", fit_bfm, bfm_full, bfm_valid, train, test, shape)]
    both = tuple(np.concatenate(ab) for ab in zip(train, test))
    sec = platform_time(fit_bfm, bfm_full, both, shape)
    save_csv([dict(method="Bayesian FM", hyperparams=results[1]["hyperparams"], n_ratings=len(both[2]),
                   threads=THREADS, fit_complete_time_s=round(sec, 1))], "bayes_platform_time.csv")
    print(f"Bayesian FM: fit on train+test + complete = {sec:.0f}s", flush=True)
    save_csv(results, "bayes_results.csv")
    print(f"total time {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
