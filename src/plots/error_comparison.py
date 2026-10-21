"""Ce que corrigent la normalisation par user et le boosting : RMSE par segment de la soumission 3
(FM brute 5 chaînes + kNN) et du modèle final (FM normalisée + FM brute + kNN, boosting), à partir
des sorties de src/experiments/error_analysis.py (--model fm_knn puis --model final).

Lancer depuis la racine du repo :  python -m src.plots.error_comparison
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation import FIGURES_DIR, read_csv, write_csv

BEFORE, AFTER = "#8a8984", "#2a78d6"
TEXT, TEXT_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ENGLISH = {"0 (inconnu)": "0 (unseen)", "1-1": "1", "jamais": "never"}
PANELS = [("user_std", "Spread of the user's ratings", "standard deviation of the user's ratings"),
          ("item_count", "Movie popularity", "ratings of the movie in the training folds"),
          ("user_count", "User activity", "ratings of the user in the training folds"),
          ("rating", "True rating", "true rating")]


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=TEXT, pad=8)
    ax.set_xlabel(xlabel, color=TEXT_2)
    ax.set_ylabel(ylabel, color=TEXT_2)
    ax.grid(True, color=GRID, linewidth=0.8, axis="y")
    ax.tick_params(colors=TEXT_2, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def rows_for(folder, by):
    if by == "rating":
        return [dict(segment=f"{r['rating']:g}", rmse_submission=r["rmse"], share_ratings=None)
                for r in read_csv(f"{folder}/calibration.csv")]
    return [r for r in read_csv(f"{folder}/segments.csv") if r["by"] == by]


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    table = []
    for ax, (by, title, xlabel) in zip(axes.flat, PANELS):
        before, after = rows_for("error_analysis", by), rows_for("error_analysis_final", by)
        x = np.arange(len(before))
        for rows, color, label in [(before, BEFORE, "submission 3: FM (5 chains) + kNN, CV 0.8098"),
                                   (after, AFTER, "final: normalized FM + FM + kNN + boosting, CV 0.8023")]:
            ax.plot(x, [r["rmse_submission"] for r in rows], marker="o", color=color, linewidth=2, markersize=6,
                    markeredgecolor="white", markeredgewidth=1.5, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels([ENGLISH.get(str(r["segment"]), str(r["segment"])) +
                            (f"\n{100 * r['share_ratings']:.0f}%" if r["share_ratings"] is not None else "")
                            for r in before], fontsize=8)
        style(ax, title, xlabel, "RMSE")
        for b, a in zip(before, after):
            table.append(dict(by=by, segment=b["segment"], rmse_before=b["rmse_submission"],
                              rmse_after=a["rmse_submission"], gain=round(a["rmse_submission"] - b["rmse_submission"], 4)))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("What user normalization and boosting fix (out-of-fold RMSE by segment; "
                 "% = share of ratings)", x=0.01, ha="left", y=1.06, fontsize=12, color=TEXT)
    fig.tight_layout()
    out = FIGURES_DIR / "error_analysis_final" / "comparison_by_segment.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    write_csv("error_analysis_final/comparison_by_segment.csv", table)
    for r in table:
        print(f"  {r['by']:<11} {str(r['segment']):<12} {r['rmse_before']:.4f} -> {r['rmse_after']:.4f}  ({r['gain']:+.4f})")
    print(f"Figure : {out}")


if __name__ == "__main__":
    main()
