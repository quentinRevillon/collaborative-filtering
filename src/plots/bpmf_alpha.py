"""RMSE de validation croisée de BPMF en fonction de alpha, à partir de
results/bpmf_alpha_grid.csv (produit par src/experiments/bpmf_alpha.py), avec les
références biais seuls et kNN films (même CV 5 plis).

Lancer depuis la racine du repo :  python -m src.plots.bpmf_alpha
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation import FIGURES_DIR, best_row, read_csv

OUT = FIGURES_DIR / "bpmf_alpha.png"
SERIES, TEXT, TEXT_2, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"
PAPER_ALPHA = 2


def main():
    rows = sorted(read_csv("bpmf_alpha_grid.csv"), key=lambda r: r["alpha"])
    alphas = [r["alpha"] for r in rows]
    scores = [r["cv_rmse"] for r in rows]
    best = min(rows, key=lambda r: r["cv_rmse"])
    references = {"biases only": best_row("baselines_grid.csv", "both")["cv_rmse"]}
    knn = [r for r in read_csv("knn_grid.csv") if r["kind"] == "item"]
    references["item kNN"] = min(r["cv_rmse"] for r in knn)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("white")
    for name, value in references.items():
        ax.axhline(value, color=TEXT_2, linewidth=1, linestyle="--", zorder=1)
        ax.text(alphas[-1], value, f"{name} {value:.4f}", ha="right", va="bottom",
                fontsize=9, color=TEXT_2)
    ax.plot(alphas, scores, color=SERIES, linewidth=2, marker="o", markersize=7,
            markeredgecolor="white", markeredgewidth=2, zorder=3)

    for alpha, label, offset, ha in [(best["alpha"], "CV optimum", (-14, -14), "right"),
                                     (PAPER_ALPHA, "paper", (12, -4), "left")]:
        score = next(r["cv_rmse"] for r in rows if r["alpha"] == alpha)
        ax.annotate(f"α = {alpha:g} ({label})\n{score:.4f}", (alpha, score), xytext=offset,
                    textcoords="offset points", ha=ha, va="top", fontsize=9, color=TEXT)

    ax.set_xscale("log")
    ticks = [0.5, 1, 1.5, 2, 3, 5, 8]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{a:g}" for a in ticks])
    ax.minorticks_off()
    ax.set_xlabel("α (noise precision, log scale)", color=TEXT_2)
    ax.set_ylabel("5-fold cross-validation RMSE", color=TEXT_2)
    ax.set_ylim(min(scores) - 0.008, max(scores) + 0.004)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT_2, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.set_title("BPMF with biases (rank 10): RMSE vs α", loc="left", fontsize=12, color=TEXT, pad=12)

    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Figure ecrite : {OUT}")


if __name__ == "__main__":
    main()
