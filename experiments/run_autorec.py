"""AutoRec experiments: I/U orientation, raw vs bias residuals, hidden size, L2, kernel; blend.

Usage:
    python experiments/run_autorec.py

Protocol: every configuration is trained on 90 % of the train ratings with early
stopping on the held-out 10 % (src.data.split, seed 0). The best configuration of
each family (orientation x target, plus the kernelized variant) is refitted on the
full train set for its selected number of epochs and scored once on the test set.
Selection uses the validation RMSE only. Finally the best AutoRec is added to the
ridge blend of run_alternatives.py (models refitted with the hyperparameters
stored in results/alt_results.csv) to measure the blend gain.

Writes results/autorec_grid.csv, results/autorec_results.csv,
results/autorec_curves.csv, results/autorec_blend.csv and figures/autorec/*.png.
Runs in about 10-15 min on 4 CPU threads.
"""
import csv
import itertools
import json
import sys
import time
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run as a script from the repo root
from src.autorec import fit_autorec, fit_autorec_early
from src.baselines import fit_biases
from src.blend import combine, cv_rmse, fit_all, learn_weights
from src.data import evaluate, load_coo, split
from src.knn import fit_item_knn
from src.mf import fit_als
from src.side_info import fit_side_biases, item_features
from src.soft_impute import fit_soft_impute

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figures" / "autorec"
LAMS = [10.0, 30.0, 100.0, 300.0]
HIDDENS = [100, 300, 500]
KERNEL_DIM = 5
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3})


def save_csv(rows, name):
    """Write a list of dicts as results/<name>."""
    with open(RES / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] {RES / name}")


def save_fig(fig, name):
    """Save a figure in figures/autorec/ and close it."""
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {FIG / name}")


def family(hp):
    """Readable name of a configuration family, e.g. 'I-AutoRec residual (kernel)'."""
    name = f"{'I' if hp['item'] else 'U'}-AutoRec {'residual' if hp['residual'] else 'raw'}"
    return name + (" (kernel)" if hp["kernel_dim"] else "")


def run_config(hp, tr, va, shape):
    """Early-stopped training of one configuration; returns (grid row, curves, predict)."""
    t = time.perf_counter()
    predict, curves = fit_autorec_early(tr, shape, va, **hp)
    k = int(np.argmin(curves[:, 1]))
    row = dict(family=family(hp), **hp, epochs=k + 1, train_rmse=round(curves[k, 0], 4),
               valid_rmse=round(curves[k, 1], 4), time_s=round(time.perf_counter() - t, 1))
    print(f"  {row}", flush=True)
    return row, curves, predict


def run_grid(hps, tr, va, shape, runs):
    """Run every configuration not already in `runs` ({key: (row, curves, predict)})."""
    for hp in hps:
        key = json.dumps(hp)
        runs[key] = runs[key] if key in runs else run_config(hp, tr, va, shape)


def config(item, residual, hidden=500, lam=30.0, kernel_dim=0):
    """Hyperparameter dict of fit_autorec (without epochs)."""
    return dict(hidden=hidden, lam=lam, item=item, residual=residual, kernel_dim=kernel_dim)


def best_of(runs, name):
    """Run (row, curves, predict) with the lowest valid RMSE within a family."""
    return min((r for r in runs.values() if r[0]["family"] == name), key=lambda r: r[0]["valid_rmse"])


def refit(row, train, test, shape):
    """Refit a selected configuration on the full train set and score it once on test."""
    hp = {k: row[k] for k in ("hidden", "lam", "item", "residual", "kernel_dim", "epochs")}
    t = time.perf_counter()
    predict = fit_autorec(train, shape, **hp)
    fit_time = time.perf_counter() - t
    r, a = evaluate(predict, test)
    print(f"{row['family']:30s} valid={row['valid_rmse']:.4f} test={r:.4f} acc={a:.4f} fit={fit_time:.1f}s", flush=True)
    return dict(method=row["family"], hyperparams=json.dumps(hp), valid_rmse=row["valid_rmse"],
                test_rmse=round(r, 4), test_acc=round(a, 4), fit_time_s=round(fit_time, 2)), predict


def alt_fits(F):
    """{name: fit(train, shape)} of the run_alternatives models with their tuned hyperparameters."""
    fns = {"biases": fit_biases, "side-info biases": partial(fit_side_biases, F=F), "soft-impute": fit_soft_impute,
           "item-kNN": fit_item_knn, "MF-ALS": fit_als}
    with open(RES / "alt_results.csv") as f:
        hps = {r["method"]: json.loads(r["hyperparams"]) for r in csv.DictReader(f) if r["method"] in fns}
    return {n: partial(fns[n], **hps[n]) for n in fns}


def blend_row(name, part, full, va, test):
    """Fit the ridge blend (alpha by 5-fold CV on valid) and score it on test; returns (row, ridge)."""
    alpha = min(ALPHAS, key=lambda al: cv_rmse(part, va, al))
    ridge = learn_weights(part, va, alpha)
    r, a = evaluate(combine(full, ridge), test)
    print(f"{name:30s} valid(5-fold CV)={cv_rmse(part, va, alpha):.4f} test={r:.4f} acc={a:.4f}", flush=True)
    weights = {n: round(float(w), 4) for n, w in zip(part, ridge.coef_)}
    return dict(method=name, alpha=alpha, valid_cv_rmse=round(cv_rmse(part, va, alpha), 4),
                test_rmse=round(r, 4), test_acc=round(a, 4), weights=json.dumps(weights)), ridge


def plot_curves(selected, bias_valid):
    """Train / valid RMSE vs epoch for the best configuration of each family."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), sharex=False)
    for c, (row, curves, _) in enumerate(selected):
        x = np.arange(1, len(curves) + 1)
        lab = f"{row['family']} (h={row['hidden']}, λ={row['lam']:g})"
        a1.plot(x, curves[:, 0], color=COLORS[c], lw=2, label=lab)
        a2.plot(x, curves[:, 1], color=COLORS[c], lw=2, label=lab)
        a2.plot(row["epochs"], row["valid_rmse"], marker=MARKERS[c], color=COLORS[c], ms=8)
    a2.axhline(bias_valid, color="#777", ls="--", lw=1.5, label="biais seuls")
    a1.set(xlabel="époque (pas Adam full-batch)", ylabel="RMSE train", title="AutoRec : RMSE train")
    a2.set(xlabel="époque (pas Adam full-batch)", ylabel="RMSE validation", ylim=(0.88, 1.0),
           title="AutoRec : RMSE validation (point = arrêt précoce)")
    a2.legend(frameon=False, fontsize=8)
    save_fig(fig, "learning_curves.png")


def plot_lambda_by_family(rows, bias_valid):
    """Valid RMSE vs lambda (hidden 500, no kernel), one curve per orientation x target."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    names = sorted({r["family"] for r in rows if r["hidden"] == 500 and not r["kernel_dim"]})
    for c, name in enumerate(names):
        pts = sorted((r["lam"], r["valid_rmse"]) for r in rows
                     if r["family"] == name and r["hidden"] == 500)
        ax.plot(*zip(*pts), marker=MARKERS[c], color=COLORS[c], lw=2, ms=6, label=name)
    ax.axhline(bias_valid, color="#777", ls="--", lw=1.5, label="biais seuls")
    ax.set(xscale="log", xlabel="pénalité L2 λ", ylabel="RMSE validation (arrêt précoce)",
           title="AutoRec (500 unités) : orientation et cible")
    ax.legend(frameon=False, fontsize=9)
    save_fig(fig, "rmse_vs_lambda_variants.png")


def plot_hidden(rows, name, bias_valid):
    """Valid RMSE vs lambda for each hidden size, within the selected family."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for c, h in enumerate(HIDDENS):
        pts = sorted((r["lam"], r["valid_rmse"]) for r in rows if r["family"] == name and r["hidden"] == h)
        ax.plot(*zip(*pts), marker=MARKERS[c], color=COLORS[c], lw=2, ms=6, label=f"{h} unités")
    ax.axhline(bias_valid, color="#777", ls="--", lw=1.5, label="biais seuls")
    ax.set(xscale="log", xlabel="pénalité L2 λ", ylabel="RMSE validation (arrêt précoce)",
           title=f"{name} : taille de la couche cachée")
    ax.legend(frameon=False)
    save_fig(fig, "rmse_vs_hidden.png")


def curve_rows(selected):
    """Long-format rows (family, epoch, train, valid RMSE) of the selected learning curves."""
    return [dict(family=row["family"], epoch=e + 1, train_rmse=round(t, 4), valid_rmse=round(v, 4))
            for row, curves, _ in selected for e, (t, v) in enumerate(curves)]


def main():
    """Tune, refit and score AutoRec variants, then test them in the ridge blend."""
    RES.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    train, shape = load_coo(ROOT / "data" / "ratings_train.npy")
    test, _ = load_coo(ROOT / "data" / "ratings_test.npy")
    tr, va = split(train, 0.1, seed=0)
    bias_valid = evaluate(fit_biases(tr, shape), va)[0]
    runs = {}

    print("stage 1: orientation x target (hidden 500)", flush=True)
    run_grid([config(it, rs, lam=lam) for it, rs, lam in itertools.product([True, False], [False, True], LAMS)],
             tr, va, shape, runs)
    top = min((r[0] for r in runs.values()), key=lambda r: r["valid_rmse"])
    print(f"stage 2: hidden size for {top['family']}", flush=True)
    run_grid([config(top["item"], top["residual"], h, lam) for h, lam in itertools.product(HIDDENS, LAMS)],
             tr, va, shape, runs)
    top = min((r[0] for r in runs.values()), key=lambda r: r["valid_rmse"])
    print("stage 3: kernelized weights (Sparse FC)", flush=True)
    run_grid([config(top["item"], top["residual"], top["hidden"], lam, KERNEL_DIM) for lam in LAMS[:3]],
             tr, va, shape, runs)
    rows = [r[0] for r in runs.values()]
    save_csv(rows, "autorec_grid.csv")

    names = list(dict.fromkeys(r["family"] for r in rows))
    selected = [best_of(runs, n) for n in names]
    save_csv(curve_rows(selected), "autorec_curves.csv")
    plot_curves(selected, bias_valid)
    plot_lambda_by_family(rows, bias_valid)
    plot_hidden(rows, top["family"], bias_valid)

    print("refit on full train, single test evaluation", flush=True)
    results = [refit(s[0], train, test, shape) for s in selected]
    save_csv([r for r, _ in results], "autorec_results.csv")

    print("blend", flush=True)
    best = min(selected, key=lambda s: s[0]["valid_rmse"])
    full_best = results[selected.index(best)][1]
    fits = alt_fits(item_features(ROOT / "data" / "namesngenre.npy"))
    part, full = fit_all(fits, tr, shape), fit_all(fits, train, shape)
    without, _ = blend_row("blend (ridge, alt models)", part, full, va, test)
    part["AutoRec"], full["AutoRec"] = best[2], full_best
    with_ar, _ = blend_row(f"blend + {best[0]['family']}", part, full, va, test)
    save_csv([without, with_ar], "autorec_blend.csv")
    print(f"total time {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
