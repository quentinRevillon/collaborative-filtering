"""Courbes de convergence GD vs ALS à partir de results/gd_vs_als_curves.csv
(produit par src/experiments/als.py), en fonction du temps de calcul.

    gauche : excès de coût (C - C*) / C*, échelle log, C* = plus petit coût atteint
    droite : RMSE de validation, avec la baseline « biais seuls » en référence

Lancer depuis la racine du repo :  python -m src.plots.convergence
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation import FIGURES_DIR, read_csv

OUT = FIGURES_DIR / "gd_vs_als_convergence.png"
COLORS = {"GD": "#2a78d6", "ALS": "#eb6834"}
LABELS = {"GD": "Gradient descent (momentum 0.9)", "ALS": "ALS"}
TEXT, TEXT_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_curves():
    """Renvoie ({méthode: courbes}, RMSE de validation de la baseline biais)."""
    runs, baseline_val = {}, None
    for row in read_csv("gd_vs_als_curves.csv"):
        if row["method"] == "baseline":
            baseline_val = row["val_rmse"]
            continue
        run = runs.setdefault(row["method"], {"time": [], "loss": [], "val": []})
        run["time"].append(row["time"])
        run["loss"].append(row["loss"])
        run["val"].append(row["val_rmse"])
    return runs, baseline_val


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=TEXT, pad=10)
    ax.set_xlabel("compute time (s)", color=TEXT_2)
    ax.set_ylabel(ylabel, color=TEXT_2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT_2, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def main():
    runs, baseline_val = load_curves()
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
        ax_val.annotate(f"{name}: {run['val'][best]:.4f}\n(it. {best + 1}, {run['time'][best]:.1f} s)",
                        (run["time"][best], run["val"][best]),
                        xytext=(0, -34) if name == "ALS" else (14, -34),
                        textcoords="offset points", ha="center" if name == "ALS" else "left",
                        fontsize=9, color=TEXT)

    ax_loss.set_yscale("log")
    style(ax_loss, "Cost C: relative gap to the minimum", "(C − C*) / C*  (log)")

    ax_val.axhline(baseline_val, color=TEXT_2, linewidth=1, linestyle="--")
    ax_val.text(ax_val.get_xlim()[1], baseline_val, f"biases only {baseline_val}  ",
                ha="right", va="top", fontsize=9, color=TEXT_2)
    best_val = min(min(run["val"]) for run in runs.values())
    ax_val.set_ylim(best_val - 0.01, baseline_val + 0.04)
    style(ax_val, "Validation RMSE", "RMSE")

    handles, labels = ax_loss.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=10,
               labelcolor=TEXT, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("MF with biases: GD vs ALS, same configuration (CV fold 1)",
                 x=0.01, ha="left", y=1.08, fontsize=13, color=TEXT)
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Figure écrite : {OUT}")


if __name__ == "__main__":
    main()
