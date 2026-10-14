"""Courbes de convergence GD vs ALS à partir de results/gd_vs_als_curves.csv
(produit par src/run_als.py), en fonction du temps de calcul.

    gauche : excès de coût (C - C*) / C*, échelle log, C* = plus petit coût atteint
    droite : RMSE de validation, avec la baseline « biais seuls » en référence

Lancer depuis la racine du repo :  python -m src.plot_convergence
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CURVES = ROOT / "results" / "gd_vs_als_curves.csv"
OUT = ROOT / "figures" / "gd_vs_als_convergence.png"

BASELINE_VAL = 0.8789  # biais régularisés lam=2 sur la même validation (src/run_baselines.py)
COLORS = {"GD": "#2a78d6", "ALS": "#eb6834"}
LABELS = {"GD": "Descente de gradient (momentum 0.9)", "ALS": "ALS"}
TEXT, TEXT_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_curves():
    runs = {}
    with open(CURVES) as f:
        for row in csv.DictReader(f):
            run = runs.setdefault(row["method"], {"time": [], "loss": [], "val": []})
            run["time"].append(float(row["time"]))
            run["loss"].append(float(row["loss"]))
            run["val"].append(float(row["val_rmse"]))
    return runs


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=TEXT, pad=10)
    ax.set_xlabel("temps de calcul (s)", color=TEXT_2)
    ax.set_ylabel(ylabel, color=TEXT_2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT_2, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def main():
    runs = load_curves()
    c_star = min(min(run["loss"]) for run in runs.values())

    fig, (ax_loss, ax_val) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.patch.set_facecolor("white")

    for name, run in runs.items():
        color = COLORS[name]
        excess = [(c - c_star) / c_star for c in run["loss"]]
        pts = [(t, e) for t, e in zip(run["time"], excess) if e > 0]
        ax_loss.plot(*zip(*pts), color=color, linewidth=2, label=LABELS[name])

        ax_val.plot(run["time"], run["val"], color=color, linewidth=2, label=LABELS[name])
        best = min(range(len(run["val"])), key=run["val"].__getitem__)
        ax_val.plot(run["time"][best], run["val"][best], "o", markersize=8, color=color,
                    markeredgecolor="white", markeredgewidth=2, zorder=3)
        ax_val.annotate(f"{name} : {run['val'][best]:.4f}\n(it. {best + 1}, {run['time'][best]:.1f} s)",
                        (run["time"][best], run["val"][best]),
                        xytext=(0, -34) if name == "ALS" else (14, -34),
                        textcoords="offset points", ha="center" if name == "ALS" else "left",
                        fontsize=9, color=TEXT)

    ax_loss.set_yscale("log")
    style(ax_loss, "Coût C : écart relatif au minimum", "(C − C*) / C*  (log)")

    ax_val.axhline(BASELINE_VAL, color=TEXT_2, linewidth=1, linestyle="--")
    ax_val.text(ax_val.get_xlim()[1], BASELINE_VAL, f"biais seuls {BASELINE_VAL}  ",
                ha="right", va="bottom", fontsize=9, color=TEXT_2)
    ax_val.set_ylim(0.865, 0.92)
    style(ax_val, "RMSE de validation", "RMSE")

    handles, labels = ax_loss.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=10,
               labelcolor=TEXT, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("MF avec biais (k=50, λ=10) : même minimum, ALS ~5× plus rapide",
                 x=0.01, ha="left", y=1.08, fontsize=13, color=TEXT)
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Figure écrite : {OUT}")


if __name__ == "__main__":
    main()
