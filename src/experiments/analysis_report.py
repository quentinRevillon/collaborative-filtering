"""Analyse pour le rapport, dans l'ordre du raisonnement (prédictions hors pli, CV 5 plis sur
train + test ; aucun réentraînement) :

1. Modèle de base : Bayesian FM, moyenne de 5 chaînes, notes brutes.
   1a. Notes extrêmes et calibration : la régression vers la moyenne est imposée par la RMSE.
       Pour un prédicteur calibré (E[r | r̂] = r̂), la pente de r̂ en fonction de r vaut
       Var(r̂) / Var(r) = 1 - MSE / Var(r) : elle mesure la part de variance expliquée.
       Test : étirer les prédictions autour de leur moyenne, r̂' = m + g (r̂ - m), pour g autour de 1.
   1b. Users dont les notes varient beaucoup : erreur par dispersion de l'user, FM brute vs FM sur
       notes normalisées par user (src/models/normalized.py).
   1c. Films peu notés : erreur par popularité, biais / kNN / FM brute / FM normalisée.
2. Combinaison FM + kNN films (stack linéaire, poids appris sur 4 plis, évalué sur le 5e) :
   poids, apport du kNN par segment, corrélation des erreurs, pour la FM brute puis normalisée.
Sorties : results/analysis_report/*.csv, figures/analysis_report/*.png (texte en anglais).

Lancer depuis la racine du repo :  python -m src.experiments.analysis_report
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation import FIGURES_DIR, RESULTS_DIR, write_csv
from src.experiments.error_analysis import buckets, load_oof

OUT = "analysis_report"
FIG = FIGURES_DIR / OUT
LEVELS = np.arange(0.5, 5.01, 0.5)
GRAY, ORANGE, AQUA, BLUE, VIOLET = "#8a8984", "#eb6834", "#1baf7a", "#2a78d6", "#4a3aa7"
TEXT, TEXT_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def rmse(e):
    return float(np.sqrt(np.mean(e ** 2)))


def clip(p):
    return np.clip(p, 0.5, 5.0)


def per_fold(pred, y, fold):
    return np.array([rmse(y[fold == k] - clip(pred[fold == k])) for k in range(fold.max() + 1)])


def stack(X, y, fold):
    """Stack linéaire appris sur 4 plis, évalué sur le 5e ; renvoie (prédiction, poids appris sur tout)."""
    pred = np.empty_like(y)
    for k in range(fold.max() + 1):
        tr = fold != k
        w = np.linalg.lstsq(np.column_stack([np.ones(tr.sum()), X[tr]]), y[tr], rcond=None)[0]
        pred[~tr] = w[0] + X[~tr] @ w[1:]
    w_all = np.linalg.lstsq(np.column_stack([np.ones(len(y)), X]), y, rcond=None)[0]
    return pred, w_all


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


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def line(ax, x, values, color, label):
    ax.plot(x, values, marker="o", color=color, linewidth=2, markersize=7, markeredgecolor="white",
            markeredgewidth=2, label=label)


def segment_rmse(segs, preds, y):
    rows = []
    for name, m in segs:
        row = dict(segment=name, n=int(m.sum()), share_ratings=round(m.mean(), 4))
        row.update({f"rmse_{k}": round(rmse(y[m] - clip(p[m])), 4) for k, p in preds.items()})
        rows.append(row)
    return rows


def main():
    oof = load_oof()
    y, fold = oof["truth"], oof["fold_id"]
    norm_chains = np.load(RESULTS_DIR / "final_stack_train.npz")["fm_norm_chains"].astype(float)
    fm = np.mean([oof[f"fm_{s}"] for s in range(5)], axis=0)          # FM x5, notes brutes
    fm_norm = norm_chains.mean(0)                                     # FM x5, notes normalisées
    knn, biases = oof["knn_films"], oof["biais"]

    print("Vue d'ensemble (RMSE CV 5 plis sur train + test) :")
    overview = {"biases only": biases, "item kNN": knn, "Bayesian FM x5 (raw ratings)": fm,
                "Bayesian FM x5 (user-normalized)": fm_norm}
    rows = []
    for name, p in overview.items():
        s = per_fold(p, y, fold)
        rows.append(dict(model=name, cv_rmse=round(s.mean(), 4), cv_std=round(s.std(), 4)))
        print(f"  {name:<36} {s.mean():.4f} +/- {s.std():.4f}")
    write_csv(f"{OUT}/overview.csv", rows)

    # ------------------------------------------------------------------ 1a
    print("\n1a. FM x5 brute : notes extremes et calibration")
    p = clip(fm)
    e = y - p
    var_r, mse = y.var(), np.mean(e ** 2)
    slope_fwd = np.polyfit(y, p, 1)[0]          # r̂ en fonction de r
    slope_rev = np.polyfit(p, y, 1)[0]          # r en fonction de r̂ (calibration)
    print(f"  Var(r) = {var_r:.4f}, MSE = {mse:.4f}, 1 - MSE/Var(r) = {1 - mse / var_r:.3f}")
    print(f"  pente de r_hat en fonction de r = {slope_fwd:.3f}  (regression vers la moyenne)")
    print(f"  pente de r en fonction de r_hat = {slope_rev:.3f}  (calibration : 1 = bien calibre)")
    print(f"  Var(r_hat) / Var(r) = {p.var() / var_r:.3f}")
    calib = []
    for lv in LEVELS:
        m = y == lv
        calib.append(dict(rating=lv, n=int(m.sum()), mean_pred=round(p[m].mean(), 3), rmse=round(rmse(e[m]), 4),
                          share_sse=round((e[m] ** 2).sum() / (e ** 2).sum(), 4)))
    write_csv(f"{OUT}/fm_by_true_rating.csv", calib)
    edges = np.arange(0.5, 5.01, 0.25)
    reliab = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 5 else (p >= lo)
        if m.sum() >= 50:
            reliab.append(dict(mean_pred=p[m].mean(), mean_true=y[m].mean(), n=int(m.sum())))
    write_csv(f"{OUT}/fm_reliability.csv", reliab)
    # étirement des prédictions : r̂' = m + g (r̂ - m) ; RMSE en fonction de g
    gammas = np.round(np.arange(0.7, 1.51, 0.05), 2)
    center = fm.mean()
    stretch = [dict(gamma=g, cv_rmse=round(per_fold(center + g * (fm - center), y, fold).mean(), 5)) for g in gammas]
    best_g = min(stretch, key=lambda r: r["cv_rmse"])
    print(f"  etirement des predictions : meilleur g = {best_g['gamma']} (RMSE {best_g['cv_rmse']}), "
          f"g = 1 : {next(r['cv_rmse'] for r in stretch if r['gamma'] == 1.0)}, "
          f"g = 1.3 : {next(r['cv_rmse'] for r in stretch if r['gamma'] == 1.3)}")
    write_csv(f"{OUT}/fm_stretch.csv", stretch)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    parts = ax.boxplot([p[y == lv] for lv in LEVELS], positions=LEVELS, widths=0.3, showfliers=False,
                       patch_artist=True)
    for b in parts["boxes"]:
        b.set(facecolor="#d6e4f7", edgecolor=BLUE)
    for k in ("whiskers", "caps"):
        for ln in parts[k]:
            ln.set(color=BLUE)
    for ln in parts["medians"]:
        ln.set(color=BLUE, linewidth=2)
    ax.plot([0.5, 5], [0.5, 5], color=TEXT_2, linestyle="--", linewidth=1, label="prediction = true rating")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_xticks(LEVELS)
    ax.set_xticklabels([f"{v:g}" for v in LEVELS])
    ax.set_ylim(0.3, 5.2)
    style(ax, f"(a) Prediction by true rating (slope {slope_fwd:.2f})", "true rating", "predicted rating")
    ax = axes[1]
    ax.plot([1, 5], [1, 5], color=TEXT_2, linestyle="--", linewidth=1, label="perfect calibration")
    line(ax, [r["mean_pred"] for r in reliab], [r["mean_true"] for r in reliab], BLUE,
         "mean true rating per prediction bin")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax, f"(b) True rating by prediction (slope {slope_rev:.2f})", "predicted rating", "mean true rating")
    ax = axes[2]
    line(ax, gammas, [r["cv_rmse"] for r in stretch], BLUE, None)
    ax.axvline(1.0, color=TEXT_2, linestyle="--", linewidth=1)
    ax.text(1.01, max(r["cv_rmse"] for r in stretch), "  g = 1 (no stretch)", fontsize=8, color=TEXT_2, va="top")
    style(ax, "(c) Stretching the predictions: m + g (r̂ − m)", "stretch factor g", "RMSE (5-fold CV)")
    fig.suptitle("Bayesian FM x5: the shrinkage towards the mean is calibrated, stretching it hurts the RMSE",
                 x=0.01, ha="left", y=1.03, fontsize=12, color=TEXT)
    save(fig, "1a_extremes_calibration.png")

    # ------------------------------------------------------------------ 1b
    print("\n1b. Users dont les notes varient beaucoup : FM x5 brute vs normalisee")
    preds = {"fm_raw": fm, "fm_norm": fm_norm}
    segs = buckets(oof["std_u"], [0, 0.6, 0.8, 1.0, 1.2, 10], lambda lo, hi: f"{lo:.1f}-{hi:.1f}" if hi < 10 else f">= {lo:.1f}")
    rows_std = segment_rmse(segs, preds, y)
    sse_raw = (y - clip(fm)) ** 2
    for r, (_, m) in zip(rows_std, segs):
        r["share_sse_raw"] = round(sse_raw[m].sum() / sse_raw.sum(), 4)
        r["gain"] = round(r["rmse_fm_norm"] - r["rmse_fm_raw"], 4)
        print(f"  ecart-type {r['segment']:<9} {100 * r['share_ratings']:4.0f}% des notes, "
              f"{100 * r['share_sse_raw']:4.0f}% de l'erreur | brute {r['rmse_fm_raw']:.4f} -> normalisee "
              f"{r['rmse_fm_norm']:.4f} ({r['gain']:+.4f})")
    d = per_fold(fm_norm, y, fold) - per_fold(fm, y, fold)
    print(f"  global : {per_fold(fm, y, fold).mean():.4f} -> {per_fold(fm_norm, y, fold).mean():.4f} "
          f"(par pli : {' '.join(f'{v:+.4f}' for v in d)})")
    write_csv(f"{OUT}/fm_by_user_spread.csv", rows_std)

    # ------------------------------------------------------------------ 1c
    print("\n1c. Films peu notes")
    preds_c = {"biases": biases, "knn": knn, "fm_raw": fm, "fm_norm": fm_norm}
    segs_i = buckets(oof["n_i"], [0, 1, 2, 6, 21, 51, 10 ** 6],
                     lambda lo, hi: "0 (unseen)" if hi == 1 else ("1" if hi == 2 else f"{lo}-{hi - 1}" if hi < 10 ** 6 else f">= {lo}"))
    rows_pop = segment_rmse(segs_i, preds_c, y)
    for r in rows_pop:
        print(f"  {r['segment']:<11} {100 * r['share_ratings']:4.0f}% | biais {r['rmse_biases']:.4f}  kNN {r['rmse_knn']:.4f}"
              f"  FM brute {r['rmse_fm_raw']:.4f}  FM normalisee {r['rmse_fm_norm']:.4f}")
    write_csv(f"{OUT}/by_movie_popularity.csv", rows_pop)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(rows_std))
    for key, color, label in [("fm_raw", AQUA, "FM x5, raw ratings"), ("fm_norm", BLUE, "FM x5, user-normalized")]:
        line(axes[0], x, [r[f"rmse_{key}"] for r in rows_std], color, label)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{r['segment']}\n{100 * r['share_ratings']:.0f}% of ratings,\n"
                             f"{100 * r['share_sse_raw']:.0f}% of the error" for r in rows_std], fontsize=8)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    style(axes[0], "(a) Users whose ratings vary a lot", "standard deviation of the user's ratings", "RMSE")
    x = np.arange(len(rows_pop))
    for key, color, label in [("biases", GRAY, "biases only"), ("knn", ORANGE, "item kNN"),
                              ("fm_raw", AQUA, "FM x5, raw ratings"), ("fm_norm", BLUE, "FM x5, user-normalized")]:
        line(axes[1], x, [r[f"rmse_{key}"] for r in rows_pop], color, label)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{r['segment']}\n{100 * r['share_ratings']:.0f}% of ratings" for r in rows_pop], fontsize=8)
    axes[1].legend(frameon=False, fontsize=9, loc="upper right")
    style(axes[1], "(b) Rarely rated movies", "ratings of the movie in the training folds", "RMSE")
    save(fig, "1bc_users_and_movies.png")

    # ------------------------------------------------------------------ 2
    print("\n2. Combinaison FM x5 + kNN films (stack lineaire)")
    contrib_rows, seg_rows = [], []
    segs_u = buckets(oof["n_u"], [0, 20, 50, 100, 300, 10 ** 6], lambda lo, hi: f"{lo}-{hi - 1}" if hi < 10 ** 6 else f">= {lo}")
    for label, base in [("raw", fm), ("normalized", fm_norm)]:
        pred, w = stack(np.column_stack([base, knn]), y, fold)
        s_fm, s_st = per_fold(base, y, fold), per_fold(pred, y, fold)
        e_fm, e_knn = y - clip(base), y - clip(knn)
        corr = np.corrcoef(e_fm, e_knn)[0, 1]
        # contribution de chaque modèle à l'écart de la prédiction à sa moyenne
        c_fm, c_knn = w[1] * (base - base.mean()), w[2] * (knn - knn.mean())
        share_knn = np.var(c_knn) / (np.var(c_fm) + np.var(c_knn))
        contrib_rows.append(dict(fm=label, w_intercept=round(w[0], 4), w_fm=round(w[1], 4), w_knn=round(w[2], 4),
                                 rmse_fm=round(s_fm.mean(), 4), rmse_stack=round(s_st.mean(), 4),
                                 gain=round((s_st - s_fm).mean(), 4), corr_errors=round(corr, 3),
                                 share_knn_in_prediction_variance=round(share_knn, 3)))
        print(f"  FM {label:<10} poids : intercept {w[0]:+.3f}, FM {w[1]:.3f}, kNN {w[2]:.3f} | "
              f"FM seule {s_fm.mean():.4f} -> stack {s_st.mean():.4f} ({(s_st - s_fm).mean():+.4f}) | "
              f"correlation des erreurs FM/kNN {corr:.3f} | part du kNN dans la variance de la prediction {share_knn:.1%}")
        for by, segs in [("movie popularity", segs_i), ("user activity", segs_u)]:
            for name, m in segs:
                seg_rows.append(dict(fm=label, by=by, segment=name, share_ratings=round(m.mean(), 4),
                                     rmse_fm=round(rmse(y[m] - clip(base[m])), 4),
                                     rmse_stack=round(rmse(y[m] - clip(pred[m])), 4),
                                     gain=round(rmse(y[m] - clip(pred[m])) - rmse(y[m] - clip(base[m])), 4)))
    write_csv(f"{OUT}/stack_contributions.csv", contrib_rows)
    write_csv(f"{OUT}/stack_gain_by_segment.csv", seg_rows)
    for r in seg_rows:
        if r["by"] == "movie popularity":
            print(f"  FM {r['fm']:<10} films {r['segment']:<11} FM {r['rmse_fm']:.4f} -> stack {r['rmse_stack']:.4f} ({r['gain']:+.4f})")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, by, xlabel in [(axes[0], "movie popularity", "ratings of the movie in the training folds"),
                           (axes[1], "user activity", "ratings of the user in the training folds")]:
        sub = [r for r in seg_rows if r["by"] == by]
        names = [r["segment"] for r in sub if r["fm"] == "raw"]
        x = np.arange(len(names))
        for fm_label, color, lab in [("raw", AQUA, "kNN added to FM x5 (raw)"),
                                     ("normalized", BLUE, "kNN added to FM x5 (user-normalized)")]:
            line(ax, x, [1000 * r["gain"] for r in sub if r["fm"] == fm_label], color, lab)
        ax.axhline(0, color=TEXT_2, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{n}\n{100 * r['share_ratings']:.0f}%" for n, r in
                            zip(names, [r for r in sub if r["fm"] == "raw"])], fontsize=8)
        style(ax, f"RMSE change when the item kNN is added, by {by}", xlabel, "RMSE change (x 1000)")
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("Contribution of the item kNN on top of the Bayesian FM (negative = the kNN helps)",
                 x=0.01, ha="left", y=1.03, fontsize=12, color=TEXT)
    save(fig, "2_knn_contribution.png")
    print(f"\nFigures : {FIG}")


if __name__ == "__main__":
    main()
