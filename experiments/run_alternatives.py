"""Alternative approaches: Soft-Impute, item-kNN, side-info biases, MF, blend.

Usage:
    python experiments/run_alternatives.py

Protocol: hyperparameters are tuned on a 10 % validation split of the train
ratings (seed 0); the selected configuration is refitted on the full train set
and scored once on the test set (never used for selection). Writes
results/alt_*.csv and figures/alt/*.png. Runs in a few minutes on 4 CPU threads.
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
from src.baselines import fit_biases
from src.blend import combine, cv_rmse, learn_weights
from src.data import accuracy, clip, evaluate, load_coo, rmse, split
from src.knn import fit_item_knn
from src.mf import fit_als
from src.side_info import fit_side_biases, item_features
from src.soft_impute import fit_soft_impute, fit_soft_impute_path

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figures" / "alt"
TAUS = [40.0, 30.0, 20.0, 15.0, 12.0, 10.0, 8.0, 6.0, 4.0]
EDGES = [0, 1, 3, 5, 10, 20, 50, np.inf]
LABELS = ["0", "1-2", "3-4", "5-9", "10-19", "20-49", "50+"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
RARE = 5
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3})


def grid(**axes):
    """Cartesian product of named axes as a list of dicts."""
    return [dict(zip(axes, c)) for c in itertools.product(*axes.values())]


def save_csv(rows, name):
    """Write a list of dicts as results/<name>.csv."""
    with open(RES / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] {RES / name}")


def save_fig(fig, name):
    """Save a figure in figures/alt/ and close it."""
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {FIG / name}")


def tune(fit, hps, tr, va, shape):
    """Valid RMSE of `fit` for each hp dict; returns (rows, best_hp, best_rmse)."""
    rows = [dict(hp, valid_rmse=evaluate(fit(tr, shape, **hp), va)[0]) for hp in hps]
    best = min(rows, key=lambda r: r["valid_rmse"])
    print(f"  {fit.__name__}: best {best}", flush=True)
    return rows, {k: best[k] for k in hps[0]}, best["valid_rmse"]


def tune_soft_impute(tr, va, shape, lams):
    """Valid RMSE over the tau path (warm starts) for each bias lambda."""
    rows = [dict(lam=lam, tau=tau, valid_rmse=evaluate(p, va)[0])
            for lam in lams for tau, p in fit_soft_impute_path(tr, shape, TAUS, lam).items()]
    best = min(rows, key=lambda r: r["valid_rmse"])
    print(f"  soft-impute: best {best}", flush=True)
    return rows, {"tau": best["tau"], "lam": best["lam"]}, best["valid_rmse"]


def refit(name, fit, hp, valid_rmse, train, test, shape, rare):
    """Refit on full train, score on test (all and rare items); returns (row, predict)."""
    t = time.perf_counter()
    predict = fit(train, shape, **hp)
    fit_time = time.perf_counter() - t
    r, a = evaluate(predict, test)
    row = dict(method=name, hyperparams=json.dumps(hp), valid_rmse=round(valid_rmse, 4),
               test_rmse=round(r, 4), test_acc=round(a, 4),
               test_rmse_rare=round(evaluate(predict, rare)[0], 4), fit_time_s=round(fit_time, 2))
    print(f"{name:22s} valid={valid_rmse:.4f} test={r:.4f} acc={a:.4f} fit={fit_time:.1f}s {hp}", flush=True)
    return row, predict


def subset(data, mask):
    """Restrict COO triplets to a boolean mask."""
    return tuple(a[mask] for a in data)


def item_counts(train, shape):
    """Number of train ratings of every item."""
    return np.bincount(train[1], minlength=shape[1])


def plot_soft_impute(rows, bias_valid):
    """Valid RMSE vs tau, one curve per bias lambda."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for c, lam in enumerate(sorted({r["lam"] for r in rows})):
        pts = sorted((r["tau"], r["valid_rmse"]) for r in rows if r["lam"] == lam)
        ax.plot(*zip(*pts), marker=MARKERS[c], color=COLORS[c], lw=2, ms=6, label=f"λ biais = {lam:g}")
    ax.axhline(bias_valid, color="#777", ls="--", lw=1.5, label="biais seuls (meilleur λ)")
    ax.set(xlabel="seuil τ", ylabel="RMSE validation", title="Soft-Impute : RMSE validation selon τ")
    ax.legend(frameon=False)
    save_fig(fig, "softimpute_tau.png")


def plot_knn(rows, shrink, bias_valid):
    """Valid RMSE vs k for each denominator regularization beta (best shrink)."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for c, beta in enumerate(sorted({r["beta"] for r in rows})):
        pts = sorted((r["k"], r["valid_rmse"]) for r in rows if r["beta"] == beta and r["shrink"] == shrink)
        ax.plot(*zip(*pts), marker=MARKERS[c], color=COLORS[c], lw=2, ms=6, label=f"β = {beta:g}")
    ax.axhline(bias_valid, color="#777", ls="--", lw=1.5, label="biais seuls")
    ax.set(xscale="log", xlabel="nombre de voisins k", ylabel="RMSE validation",
           title=f"Item-kNN (Pearson-baseline, shrink={shrink:g}) : RMSE selon k")
    ax.set_xticks(sorted({r["k"] for r in rows}), labels=[str(k) for k in sorted({r["k"] for r in rows})])
    ax.legend(frameon=False, title="régul. dénominateur")
    save_fig(fig, "knn_k.png")


def bucket_rmse(predicts, test, counts):
    """Test RMSE per item-popularity bucket for each model; returns rows."""
    b = np.digitize(counts[test[1]], EDGES) - 1
    rows = []
    for k, label in enumerate(LABELS):
        m = b == k
        if not m.any():
            continue
        row = dict(bucket=label, n_test=int(m.sum()))
        row.update({n: round(rmse(clip(p(test[0][m], test[1][m])), test[2][m]), 4) for n, p in predicts.items()})
        rows.append(row)
    return rows


def plot_buckets(rows, names):
    """RMSE and RMSE gain vs biases per item-popularity bucket."""
    x = np.arange(len(rows))
    ticks = [f"{r['bucket']}\n(n={r['n_test']})" for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    for c, n in enumerate(names):
        a1.plot(x, [r[n] for r in rows], marker=MARKERS[c], color=COLORS[c], lw=2, ms=6, label=n)
    a1.set(xticks=x, xticklabels=ticks, xlabel="nb de notes train de l'item", ylabel="RMSE test",
           title="RMSE test par popularité de l'item")
    a1.legend(frameon=False)
    others = [n for n in names if n != "biases"]
    w = 0.8 / len(others)
    for c, n in enumerate(others, start=1):
        gain = [r["biases"] - r[n] for r in rows]
        a2.bar(x + (c - 1 - (len(others) - 1) / 2) * w, gain, w * 0.9, color=COLORS[c], label=n)
    a2.axhline(0, color="#444", lw=1)
    a2.set(xticks=x, xticklabels=ticks, xlabel="nb de notes train de l'item",
           ylabel="gain de RMSE vs biais (>0 = mieux)", title="Où chaque méthode aide")
    a2.legend(frameon=False, fontsize=9)
    save_fig(fig, "rmse_by_popularity.png")


def plot_weights(names, ridge):
    """Horizontal bar chart of the blend weights."""
    fig, ax = plt.subplots(figsize=(6.5, 0.5 * len(names) + 1.5))
    ax.barh(names, ridge.coef_, color=COLORS[0], height=0.6)
    for y, v in enumerate(ridge.coef_):
        ax.text(v, y, f" {v:.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=10)
    ax.axvline(0, color="#444", lw=1)
    ax.set(xlabel="poids ridge", title=f"Poids du blend (intercept = {ridge.intercept_:.3f})")
    ax.invert_yaxis()
    save_fig(fig, "blend_weights.png")


def partial_round(p, alpha):
    """Move predictions a fraction alpha of the way to the nearest 0.5 step."""
    return p + alpha * (np.round(p * 2) / 2 - p)


def rounding_rows(predicts, test):
    """RMSE / accuracy of raw vs rounded-to-0.5 test predictions per model."""
    rows = []
    for n, p in predicts.items():
        raw = clip(p(test[0], test[1]))
        rnd = partial_round(raw, 1.0)
        rows.append(dict(method=n, rmse_raw=round(rmse(raw, test[2]), 4), rmse_rounded=round(rmse(rnd, test[2]), 4),
                         acc_raw=round(accuracy(raw, test[2]), 4), acc_rounded=round(accuracy(rnd, test[2]), 4)))
    return rows


def plot_rounding(predict, test):
    """Test RMSE and accuracy of the blend as predictions are pushed to 0.5 steps."""
    raw = clip(predict(test[0], test[1]))
    alphas = np.linspace(0, 1, 11)
    r = [rmse(partial_round(raw, a), test[2]) for a in alphas]
    acc = [accuracy(partial_round(raw, a), test[2]) for a in alphas]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(alphas, r, marker="o", color=COLORS[0], lw=2)
    a1.set(xlabel="α (0 = brut, 1 = arrondi à 0.5)", ylabel="RMSE test", title="Blend : RMSE vs arrondi partiel")
    a2.plot(alphas, acc, marker="s", color=COLORS[1], lw=2)
    a2.set(xlabel="α (0 = brut, 1 = arrondi à 0.5)", ylabel="accuracy test", title="Blend : accuracy vs arrondi partiel",
           ylim=(min(acc) - 0.01, max(acc) + 0.01))
    save_fig(fig, "rounding.png")


def main():
    """Tune, refit and score every alternative model; write CSVs and figures."""
    RES.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    train, shape = load_coo(ROOT / "data" / "ratings_train.npy")
    test, _ = load_coo(ROOT / "data" / "ratings_test.npy")
    F = item_features(ROOT / "data" / "namesngenre2.npy")
    tr, va = split(train, 0.1, seed=0)
    counts = item_counts(train, shape)
    rare = subset(test, counts[test[1]] < RARE)
    side = partial(fit_side_biases, F=F)
    side.__name__ = "fit_side_biases"
    results, best, part = [], {}, {}

    def add(name, fit, hp, v, grid_rows=None, grid_name=None):
        """Refit a tuned model, store its row and its 90 % / 100 % predictors."""
        row, predict = refit(name, fit, hp, v, train, test, shape, rare)
        results.append(row)
        best[name], part[name] = predict, fit(tr, shape, **hp)
        if grid_rows:
            save_csv(grid_rows, grid_name)

    g, hp, v = tune(fit_biases, grid(lam=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0]), tr, va, shape)
    add("biases", fit_biases, hp, v, g, "alt_biases_grid.csv")
    bias_valid = v

    g, hp, v = tune(side, grid(lam=[1.0, 2.0, 3.0, 5.0], lam_w=[0.1, 1.0, 10.0], pop=[False, True]), tr, va, shape)
    add("side-info biases", side, hp, v, g, "alt_side_info_grid.csv")

    g, hp, v = tune_soft_impute(tr, va, shape, lams=[1.0, 2.0, 3.0, 5.0])
    add("soft-impute", fit_soft_impute, hp, v, g, "alt_softimpute_grid.csv")
    plot_soft_impute(g, bias_valid)

    g, hp, v = tune(fit_item_knn, grid(k=[5, 10, 20, 40, 80, 160], shrink=[10.0, 100.0, 300.0],
                                       beta=[0.0, 0.2, 0.5, 1.0], lam=[2.0]), tr, va, shape)
    add("item-kNN", fit_item_knn, hp, v, g, "alt_knn_grid.csv")
    plot_knn(g, hp["shrink"], bias_valid)

    g, hp, v = tune(fit_als, grid(rank=[10], lam_f=[8.0, 12.0, 16.0], lam_b=[2.0, 3.0, 5.0]), tr, va, shape)
    add("MF-ALS", fit_als, hp, v, g, "alt_mf_grid.csv")

    alpha = min(ALPHAS, key=lambda al: cv_rmse(part, va, al))
    ridge = learn_weights(part, va, alpha)
    blend = combine(dict(best), ridge)
    r, a = evaluate(blend, test)
    v = cv_rmse(part, va, alpha)
    results.append(dict(method="blend (ridge)", hyperparams=json.dumps({"alpha": alpha, "models": list(best)}),
                        valid_rmse=round(v, 4), test_rmse=round(r, 4), test_acc=round(a, 4),
                        test_rmse_rare=round(evaluate(blend, rare)[0], 4),
                        fit_time_s=round(sum(x["fit_time_s"] for x in results), 2)))
    print(f"{'blend (ridge)':22s} valid(5-fold CV)={v:.4f} test={r:.4f} acc={a:.4f}", flush=True)
    save_csv([dict(model=n, weight=round(w, 4)) for n, w in zip(best, ridge.coef_)]
             + [dict(model="intercept", weight=round(ridge.intercept_, 4))], "alt_blend_weights.csv")
    plot_weights(list(best), ridge)
    best["blend"] = blend

    save_csv(results, "alt_results.csv")
    buckets = bucket_rmse(best, test, counts)
    save_csv(buckets, "alt_buckets.csv")
    plot_buckets(buckets, list(best))
    save_csv(rounding_rows(best, test), "alt_rounding.csv")
    plot_rounding(blend, test)
    print(f"total time {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
