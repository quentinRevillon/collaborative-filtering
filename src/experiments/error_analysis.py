"""Analyse d'erreurs de la soumission (stack Bayesian FM 5 chaînes + kNN films), sur les
prédictions hors pli de src/experiments/bfm_stacking.py (CV 5 plis sur train + test).

Aucun réentraînement : on recharge results/bfm_stacking_oof.npz, on reconstruit les plis (même
découpage) pour retrouver l'user et le film de chaque note, et on reconstitue la prédiction de
la soumission avec des poids appris sur les 4 autres plis (cross-fitting).

1. vue d'ensemble : RMSE, distribution des résidus, part des grosses erreurs ;
2. calibration : prédiction moyenne et erreur par vraie note ;
3. erreur par segment (popularité du film, activité et dispersion de l'user, genre, décennie,
   usage des demi-étoiles), avec la part de chaque segment dans l'erreur totale, et la
   comparaison FM / kNN / biais par segment ;
4. matrice de confusion (prédictions arrondies au 0.5) et exactitude ;
5. plus grosses erreurs, films et users qui pèsent le plus ;
6. pistes testées directement sur les prédictions hors pli (cross-fitting) : stacking dont les
   poids dépendent du segment (FWLS), recalibration non linéaire, stacking non linéaire
   (gradient boosting sur prédictions + caractéristiques user / film).
   La « correction du biais résiduel par user / film » est gardée pour mémoire mais son test
   est BIAISÉ : le résidu d'un user est mesuré sur les autres plis avec des modèles qui n'ont
   pas vu ces notes, alors que le modèle du pli évalué les a vues (correction comptée deux fois).
Sorties : results/error_analysis/*.csv, figures/error_analysis/*.png, log console.

Lancer depuis la racine du repo :  python -m src.experiments.error_analysis [--model fm_knn|final]
(fm_knn : soumission 3, sorties dans error_analysis/ ; final : FM normalisée + boosting, sorties
 dans error_analysis_final/)
"""
import re
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from src.data import DATA_DIR, kfold_observed, load_train_test
from src.evaluation import FIGURES_DIR, RESULTS_DIR, write_csv

OUT_RES = "error_analysis"
OUT_FIG = FIGURES_DIR / "error_analysis"
N_CHAINS = 5                      # chaînes de la soumission
LEVELS = np.arange(0.5, 5.01, 0.5)
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
TEXT, TEXT_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# --------------------------------------------------------------------------- données
def load_oof():
    """Prédictions hors pli + user, film, pli et statistiques du pli d'apprentissage de chaque note."""
    d = np.load(RESULTS_DIR / "bfm_stacking_oof.npz")
    R_train, R_test = load_train_test()
    R_all = np.where(np.isnan(R_train), R_test, R_train)
    users, items, n_u, n_i, std_u, half_u, mean_u, mean_i, std_i = [], [], [], [], [], [], [], [], []
    for R_fit, R_val in kfold_observed(R_all, n_folds=5):
        vu, vi = np.nonzero(~np.isnan(R_val))
        users.append(vu)
        items.append(vi)
        obs = ~np.isnan(R_fit)
        n_u.append(obs.sum(1)[vu])
        n_i.append(obs.sum(0)[vi])
        std_u.append(np.nanstd(R_fit, axis=1)[vu])
        half = np.nansum((R_fit * 2) % 2 == 1, axis=1) / np.maximum(obs.sum(1), 1)
        half_u.append(half[vu])
        with warnings.catch_warnings():               # films sans note dans le pli : moyenne vide
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_u.append(np.nanmean(R_fit, axis=1)[vu])
            mean_i.append(np.nan_to_num(np.nanmean(R_fit, axis=0), nan=np.nanmean(R_fit))[vi])
            std_i.append(np.nan_to_num(np.nanstd(R_fit, axis=0), nan=0.0)[vi])
    oof = {k: d[k] for k in d.files}
    oof.update(user=np.concatenate(users), item=np.concatenate(items), n_u=np.concatenate(n_u),
               n_i=np.concatenate(n_i), std_u=np.concatenate(std_u), half_u=np.concatenate(half_u),
               mean_u=np.concatenate(mean_u), mean_i=np.concatenate(mean_i), std_i=np.concatenate(std_i))
    assert np.array_equal(R_all[oof["user"], oof["item"]], oof["truth"]), "plis non reconstruits"
    return oof


def movie_meta():
    meta = np.load(DATA_DIR / "namesngenre2.npy", allow_pickle=False)
    titles, genres = meta[:, 0], [g.split("|") for g in meta[:, 1]]
    years = [re.search(r"\((\d{4})\)\s*$", t) for t in titles]
    decades = np.array([int(m.group(1)) // 10 * 10 if m else -1 for m in years])
    return titles, genres, decades


# --------------------------------------------------------------------------- outils
def lstsq(X, y):
    A = np.column_stack([np.ones(len(y)), X])
    return np.linalg.lstsq(A, y, rcond=None)[0]


def cross_fit(X, y, fold):
    """Prédiction d'un modèle linéaire y ~ 1 + X, appris sur les autres plis (cross-fitting)."""
    pred = np.empty_like(y)
    for k in np.unique(fold):
        w = lstsq(X[fold != k], y[fold != k])
        pred[fold == k] = w[0] + X[fold == k] @ w[1:]
    return np.clip(pred, 0.5, 5.0)


def rmse(e):
    return float(np.sqrt(np.mean(e ** 2))) if len(e) else float("nan")


def per_fold(pred, y, fold):
    return np.array([rmse(y[fold == k] - pred[fold == k]) for k in np.unique(fold)])


def compare(name, pred, ref_scores, y, fold, rows):
    s = per_fold(pred, y, fold)
    d = s - ref_scores
    rows.append(dict(variant=name, cv_rmse=round(s.mean(), 5), diff_vs_submission=round(d.mean(), 5),
                     folds_better=int((d < 0).sum()), per_fold_diff=" ".join(f"{v:+.4f}" for v in d)))
    print(f"  {name:<58} cv={s.mean():.4f}  ecart {d.mean():+.5f}  ({' '.join(f'{v:+.4f}' for v in d)})")


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=TEXT, pad=10)
    ax.set_xlabel(xlabel, color=TEXT_2)
    ax.set_ylabel(ylabel, color=TEXT_2)
    ax.grid(True, color=GRID, linewidth=0.8, axis="y")
    ax.tick_params(colors=TEXT_2, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def save(fig, name):
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- analyses
def segment_table(segments, preds, y, label):
    """RMSE de chaque modèle par segment, et part du segment dans l'erreur totale de la soumission."""
    total_sse = float(np.sum((y - preds["submission"]) ** 2))
    rows = []
    for name, mask in segments:
        if mask.sum() == 0:
            continue
        row = dict(by=label, segment=name, n=int(mask.sum()), share_ratings=round(mask.mean(), 4))
        for m, p in preds.items():
            row[f"rmse_{m}"] = round(rmse(y[mask] - p[mask]), 4)
        row["share_sse"] = round(float(np.sum((y[mask] - preds["submission"][mask]) ** 2)) / total_sse, 4)
        rows.append(row)
    return rows


def print_segments(rows, title):
    print(f"\n{title}")
    print(f"  {'segment':<22}{'n':>7}{'%notes':>8}{'%erreur':>9}{'stack':>8}{'FM':>8}{'kNN':>8}{'biais':>8}")
    for r in rows:
        print(f"  {str(r['segment']):<22}{r['n']:>7}{100 * r['share_ratings']:>7.1f}%{100 * r['share_sse']:>8.1f}%"
              f"{r['rmse_submission']:>8.4f}{r['rmse_fm5']:>8.4f}{r['rmse_knn']:>8.4f}{r['rmse_biais']:>8.4f}")


def buckets(values, edges, fmt):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        name = fmt(lo, hi)
        out.append((name, (values >= lo) & (values < hi)))
    return out


def final_model_oof(oof, decades, G):
    """Prédictions hors pli du modèle final (src/experiments/final_stack.py) : FM normalisée (5) +
    FM brute (5) + kNN, stack linéaire + gradient boosting, appris sur 4 plis, évalué sur le 5e."""
    from src.experiments.final_stack import boosted
    d = np.load(RESULTS_DIR / "final_stack_train.npz")
    assert np.array_equal(d["truth"], oof["truth"]), "plis differents de final_stack_train.npz"
    norm, raw = d["fm_norm_chains"].astype(float), d["fm_raw_chains"].astype(float)
    base = np.column_stack([norm.mean(0), raw.mean(0), oof["knn_films"]])
    feats = np.column_stack([base, norm.std(0), np.log1p(oof["n_i"]), np.log1p(oof["n_u"]), oof["std_u"],
                             oof["mean_u"], oof["half_u"], oof["mean_i"], oof["std_i"], decades[oof["item"]],
                             G[oof["item"]].astype(float)])
    return np.clip(boosted(base, feats, oof["truth"], oof["fold_id"]), 0.5, 5.0), norm


def main():
    global OUT_RES, OUT_FIG
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["fm_knn", "final"], default="fm_knn",
                        help="fm_knn : soumission 3 (FM brute 5 chaines + kNN) ; final : FM normalisee + boosting")
    model = parser.parse_args().model
    if model == "final":
        OUT_RES, OUT_FIG = "error_analysis_final", FIGURES_DIR / "error_analysis_final"
    oof = load_oof()
    y, fold = oof["truth"], oof["fold_id"]
    titles, genres, decades = movie_meta()
    genre_names = sorted({g for gs in genres for g in gs})
    G = np.array([[g in gs for g in genre_names] for gs in genres])
    chains = np.array([oof[f"fm_{s}"] for s in range(10)])
    fm5 = chains[:N_CHAINS].mean(0)
    X_sub = np.column_stack([fm5, oof["knn_films"]])
    if model == "final":
        sub, chains = final_model_oof(oof, decades, G)
        fm5 = chains.mean(0)                              # FM normalisée, 5 chaînes
        labels = dict(fm5="normalized Bayesian FM (5 chains)", submission="final stack (boosting)")
        title = "Modele final (FM normalisee 5 + FM brute 5 + kNN, boosting)"
    else:
        sub = cross_fit(X_sub, y, fold)                   # prédiction de la soumission 3
        labels = dict(fm5="Bayesian FM (5 chains)", submission="stack (submitted)")
        title = "Soumission 3 (FM 5 chaines + kNN films)"
    preds = {"submission": sub, "fm5": np.clip(fm5, .5, 5), "knn": np.clip(oof["knn_films"], .5, 5),
             "biais": np.clip(oof["biais"], .5, 5)}
    e = y - sub
    ref = per_fold(sub, y, fold)
    print(f"{title}, predictions hors pli sur {len(y)} notes :")
    print(f"  RMSE CV {ref.mean():.4f} | FM seule {rmse(y - preds['fm5']):.4f} | kNN seul {rmse(y - preds['knn']):.4f}"
          f" | biais seuls {rmse(y - preds['biais']):.4f}")

    # 1. vue d'ensemble -----------------------------------------------------------
    abs_e = np.abs(e)
    sse = e ** 2
    print("\n1. Residus r - r_hat :")
    print(f"  moyenne {e.mean():+.4f} | mediane {np.median(e):+.4f} | ecart-type {e.std():.4f}"
          f" | |e| median {np.median(abs_e):.3f}")
    overview = []
    for thr in [0.5, 1.0, 1.5, 2.0, 3.0]:
        m = abs_e > thr
        share = sse[m].sum() / sse.sum()
        overview.append(dict(threshold=thr, share_ratings=round(m.mean(), 4), share_sse=round(share, 4)))
        print(f"  |e| > {thr:<3}: {100 * m.mean():5.1f}% des notes, {100 * share:5.1f}% de l'erreur quadratique")
    write_csv(f"{OUT_RES}/overview.csv", overview)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(e, bins=np.arange(-4.5, 4.51, 0.1), color=BLUE, edgecolor="white", linewidth=0.3)
    style(ax, "Out-of-fold residuals of the submission (r − r̂)", "residual (stars)", "number of ratings")
    save(fig, "residuals_hist.png")

    # 2. calibration -------------------------------------------------------------
    print("\n2. Calibration par vraie note :")
    print(f"  {'note':>5}{'n':>7}{'pred moy.':>11}{'residu moy.':>13}{'RMSE':>8}{'%erreur':>9}")
    calib = []
    for lv in LEVELS:
        m = y == lv
        row = dict(rating=lv, n=int(m.sum()), mean_pred=round(sub[m].mean(), 3),
                   mean_residual=round(e[m].mean(), 3), rmse=round(rmse(e[m]), 4),
                   share_sse=round(sse[m].sum() / sse.sum(), 4))
        calib.append(row)
        print(f"  {lv:>5}{row['n']:>7}{row['mean_pred']:>11.3f}{row['mean_residual']:>+13.3f}{row['rmse']:>8.4f}"
              f"{100 * row['share_sse']:>8.1f}%")
    slope = np.polyfit(y, sub, 1)[0]
    print(f"  pente de r_hat en fonction de r : {slope:.3f} (1 = pas de regression vers la moyenne)")
    write_csv(f"{OUT_RES}/calibration.csv", calib)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    data = [sub[y == lv] for lv in LEVELS]
    parts = ax.boxplot(data, positions=LEVELS, widths=0.3, showfliers=False, patch_artist=True)
    for b in parts["boxes"]:
        b.set(facecolor="#d6e4f7", edgecolor=BLUE)
    for k in ("whiskers", "caps"):
        for line in parts[k]:
            line.set(color=BLUE)
    for line in parts["medians"]:
        line.set(color=BLUE, linewidth=2)
    ax.plot([0.5, 5], [0.5, 5], color=TEXT_2, linestyle="--", linewidth=1, label="perfect calibration")
    ax.plot([], [], color=BLUE, linewidth=2, label="median prediction (box: 25-75 %)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_xticks(LEVELS)
    ax.set_xticklabels([f"{v:g}" for v in LEVELS])
    ax.set_ylim(0.3, 5.2)
    style(ax, f"Predictions by true rating (out-of-fold, slope {slope:.2f})", "true rating", "predicted rating")
    save(fig, "calibration.png")

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = np.arange(len(LEVELS))
    ax.bar(x - 0.2, [100 * r["n"] / len(y) for r in calib], width=0.38, color=GRAY, label="share of ratings")
    ax.bar(x + 0.2, [100 * r["share_sse"] for r in calib], width=0.38, color=BLUE,
           label="share of the squared error")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in LEVELS])
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    style(ax, "Where the squared error comes from, by true rating", "true rating", "%")
    save(fig, "error_share_by_rating.png")

    # calibration dans l'autre sens : moyenne des vraies notes par niveau de prédiction
    print("\n2b. Calibration E[r | r_hat] (vraie note moyenne par tranche de prediction) :")
    edges = np.arange(0.5, 5.01, 0.25)
    reliab = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (sub >= lo) & (sub < hi) if hi < 5 else (sub >= lo) & (sub <= hi)
        if m.sum() >= 50:
            reliab.append(dict(pred_bin=f"{lo:.2f}-{hi:.2f}", n=int(m.sum()), mean_pred=round(sub[m].mean(), 3),
                               mean_true=round(y[m].mean(), 3), gap=round(y[m].mean() - sub[m].mean(), 3)))
            print(f"  {lo:.2f}-{hi:.2f}  n={m.sum():>6}  prediction moy. {sub[m].mean():.3f}  vraie moy. {y[m].mean():.3f}"
                  f"  ecart {y[m].mean() - sub[m].mean():+.3f}")
    slope_rev = np.polyfit(sub, y, 1)[0]
    print(f"  pente de r en fonction de r_hat : {slope_rev:.3f} (1 = bien calibre dans ce sens)")
    write_csv(f"{OUT_RES}/reliability.csv", reliab)
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.plot([1, 5], [1, 5], color=TEXT_2, linestyle="--", linewidth=1, label="perfect calibration")
    ax.plot([r["mean_pred"] for r in reliab], [r["mean_true"] for r in reliab], marker="o", color=BLUE, linewidth=2,
            markersize=7, markeredgecolor="white", markeredgewidth=2, label="mean true rating per prediction bin")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    style(ax, f"Calibration: true rating vs prediction (slope {slope_rev:.2f})", "predicted rating (bin mean)",
          "mean true rating")
    save(fig, "reliability.png")

    # 3. segments ----------------------------------------------------------------
    seg_rows = []
    groups = [
        ("Popularite du film (notes dans le pli d'apprentissage)",
         buckets(oof["n_i"], [0, 1, 2, 6, 21, 51, 10 ** 6],
                 lambda lo, hi: "0 (inconnu)" if hi == 1 else f"{lo}-{hi - 1}" if hi < 10 ** 6 else f">= {lo}"), "item_count"),
        ("Activite de l'user (notes dans le pli d'apprentissage)",
         buckets(oof["n_u"], [0, 20, 50, 100, 300, 10 ** 6],
                 lambda lo, hi: f"{lo}-{hi - 1}" if hi < 10 ** 6 else f">= {lo}"), "user_count"),
        ("Dispersion des notes de l'user (ecart-type)",
         buckets(oof["std_u"], [0, 0.6, 0.8, 1.0, 1.2, 10],
                 lambda lo, hi: f"{lo:.1f}-{hi:.1f}" if hi < 10 else f">= {lo:.1f}"), "user_std"),
        ("Usage des demi-etoiles par l'user (part de x.5)",
         buckets(oof["half_u"], [0, 0.01, 0.2, 0.5, 1.01],
                 lambda lo, hi: "jamais" if lo == 0 else f"{lo:.0%}-{min(hi, 1):.0%}"), "user_half"),
        ("Decennie du film",
         [(str(d) if d > 0 else "inconnue", decades[oof["item"]] == d) for d in sorted(set(decades))], "decade"),
    ]
    for title, segs, label in groups:
        rows = segment_table(segs, preds, y, label)
        print_segments(rows, f"3. {title}")
        seg_rows += rows
    rows = segment_table([(g, G[oof["item"], k]) for k, g in enumerate(genre_names)], preds, y, "genre")
    rows.sort(key=lambda r: -r["rmse_submission"])
    print_segments(rows, "3. Genre (un film peut avoir plusieurs genres : les parts se recouvrent)")
    seg_rows += rows
    write_csv(f"{OUT_RES}/segments.csv", seg_rows)

    english = {"0 (inconnu)": "0 (unseen)", "1-1": "1"}
    series = [("biais", GRAY, "biases only"), ("knn", ORANGE, "item kNN"),
              ("fm5", AQUA, labels["fm5"]), ("submission", BLUE, labels["submission"])]

    def segment_lines(ax, rows, title, xlabel):
        x = np.arange(len(rows))
        for m, color, lab in series:
            ax.plot(x, [r[f"rmse_{m}"] for r in rows], marker="o", color=color, linewidth=2, markersize=7,
                    markeredgecolor="white", markeredgewidth=2, label=lab)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{english.get(r['segment'], r['segment'])}\n{100 * r['share_ratings']:.0f}% of ratings"
                            for r in rows], fontsize=8)
        style(ax, title, xlabel, "RMSE")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    segment_lines(ax, [r for r in seg_rows if r["by"] == "item_count"],
                  "RMSE by movie popularity (out-of-fold)", "ratings of the movie in the training folds")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    save(fig, "rmse_by_item_popularity.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    segment_lines(axes[0], [r for r in seg_rows if r["by"] == "user_count"],
                  "RMSE by user activity", "ratings of the user in the training folds")
    segment_lines(axes[1], [r for r in seg_rows if r["by"] == "user_std"],
                  "RMSE by spread of the user's ratings", "standard deviation of the user's ratings")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.04))
    save(fig, "rmse_by_user.png")

    # 4. matrice de confusion ----------------------------------------------------
    rounded = np.clip(np.round(sub * 2) / 2, 0.5, 5.0)
    idx = lambda v: np.round(v * 2).astype(int) - 1
    C = np.zeros((10, 10), dtype=int)
    np.add.at(C, (idx(y), idx(rounded)), 1)
    Cn = C / C.sum(1, keepdims=True)
    acc = np.mean(rounded == y)
    within = np.mean(np.abs(rounded - y) <= 0.5)
    print(f"\n4. Exactitude (prediction arrondie au 0.5) : {acc:.3f} | a une demi-etoile pres : {within:.3f}")
    print("  part des predictions arrondies par valeur : " +
          " ".join(f"{lv:g}:{100 * np.mean(rounded == lv):.0f}%" for lv in LEVELS))
    print("  part des vraies notes par valeur          : " +
          " ".join(f"{lv:g}:{100 * np.mean(y == lv):.0f}%" for lv in LEVELS))
    write_csv(f"{OUT_RES}/confusion.csv", [dict(true=lv, **{f"pred_{p:g}": int(C[a, b]) for b, p in enumerate(LEVELS)})
                                           for a, lv in enumerate(LEVELS)])
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    cmap = LinearSegmentedColormap.from_list("blues", ["#ffffff", "#9cc2ef", BLUE, "#123a6b"])
    im = ax.imshow(Cn, cmap=cmap, vmin=0, vmax=1, origin="lower")
    for a in range(10):
        for b in range(10):
            if Cn[a, b] >= 0.01:
                ax.text(b, a, f"{100 * Cn[a, b]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if Cn[a, b] > 0.45 else TEXT)
    ax.set_xticks(range(10))
    ax.set_xticklabels([f"{v:g}" for v in LEVELS])
    ax.set_yticks(range(10))
    ax.set_yticklabels([f"{v:g}" for v in LEVELS])
    fig.colorbar(im, ax=ax, fraction=0.046, label="share of the row")
    ax.set_title("Confusion matrix (% of each true rating)", loc="left", fontsize=11, color=TEXT, pad=10)
    ax.set_xlabel("predicted rating (rounded to 0.5)", color=TEXT_2)
    ax.set_ylabel("true rating", color=TEXT_2)
    ax.tick_params(colors=TEXT_2, length=0)
    save(fig, "confusion_matrix.png")

    # 5. plus grosses erreurs ----------------------------------------------------
    print("\n5. Plus grosses erreurs (top 15) :")
    top = np.argsort(-abs_e)[:15]
    worst = []
    for n in top:
        u, i = oof["user"][n], oof["item"][n]
        row = dict(user=int(u), movie=titles[i], genres="|".join(genres[i]), true=y[n], pred=round(sub[n], 2),
                   fm5=round(fm5[n], 2), knn=round(oof["knn_films"][n], 2), movie_ratings=int(oof["n_i"][n]),
                   user_ratings=int(oof["n_u"][n]), user_std=round(oof["std_u"][n], 2))
        worst.append(row)
        print(f"  user {u:>3} ({row['user_ratings']:>4} notes, std {row['user_std']:.2f})  {titles[i][:42]:<42}"
              f" ({row['movie_ratings']:>3} notes)  vraie {y[n]:.1f}  pred {sub[n]:.2f}")
    write_csv(f"{OUT_RES}/worst_errors.csv", worst)
    by_user = np.bincount(oof["user"], sse, minlength=610)
    by_item = np.bincount(oof["item"], sse, minlength=4980)
    for label, arr, names, count in [("users", by_user, None, np.bincount(oof["user"], minlength=610)),
                                     ("films", by_item, titles, np.bincount(oof["item"], minlength=4980))]:
        order = np.argsort(-arr)
        top10 = arr[order[:10]].sum() / sse.sum()
        top_n = order[:max(1, len(arr) // 100)]
        print(f"  {label} : les 10 qui pesent le plus = {100 * top10:.1f}% de l'erreur ; "
              f"le 1% le plus lourd ({len(top_n)}) = {100 * arr[top_n].sum() / sse.sum():.1f}%")
        for j in order[:5]:
            nm = names[j][:40] if names is not None else f"user {j}"
            print(f"     {nm:<40} {count[j]:>4} notes, RMSE {np.sqrt(arr[j] / count[j]):.3f}")

    chain_std = chains[:N_CHAINS].std(0)
    if model == "fm_knn":
        improvement_tests(oof, sub, X_sub, chains, chain_std, fm5, e, abs_e, ref, y, fold, decades, G)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    q = np.quantile(chain_std, np.linspace(0, 1, 11))
    bins = np.clip(np.digitize(chain_std, q[1:-1]), 0, 9)
    ax.plot([chain_std[bins == b].mean() for b in range(10)], [rmse(e[bins == b]) for b in range(10)],
            marker="o", color=BLUE, linewidth=2, markersize=7, markeredgecolor="white", markeredgewidth=2)
    style(ax, "Error vs disagreement between the 5 FM chains", "std of the 5 chain predictions (decile mean)",
          "RMSE of the submission")
    save(fig, "error_vs_chain_disagreement.png")
    print(f"\nFigures : {OUT_FIG}")


def improvement_tests(oof, sub, X_sub, chains, chain_std, fm5, e, abs_e, ref, y, fold, decades, G):
    """Section 6 (soumission 3 uniquement) : pistes testées en cross-fitting."""
    print("\n6. Pistes testees sur les predictions hors pli (poids appris sur les autres plis) :")
    exp = []
    compare("soumission (FM5 + kNN films)", sub, ref, y, fold, exp)
    compare("FM 10 chaines + kNN films", cross_fit(np.column_stack([chains.mean(0), oof["knn_films"]]), y, fold),
            ref, y, fold, exp)
    compare("+ kNN users + biais", cross_fit(np.column_stack([X_sub, oof["knn_users"], oof["biais"]]), y, fold),
            ref, y, fold, exp)
    # FWLS : les poids dépendent de méta-features (popularité, activité, désaccord des chaînes)
    meta ={"log n_film": np.log1p(oof["n_i"]), "log n_user": np.log1p(oof["n_u"]), "std chaines": chain_std,
            "std user": oof["std_u"]}
    print(f"  (correlation |erreur| ~ desaccord des chaines : {np.corrcoef(abs_e, chain_std)[0, 1]:.3f})")
    for names in [("log n_film",), ("log n_user",), ("std chaines",), ("std user",),
                  ("log n_film", "log n_user"), ("log n_film", "log n_user", "std chaines", "std user")]:
        Z = np.column_stack([meta[m] for m in names])
        Z = (Z - Z.mean(0)) / Z.std(0)
        X = np.column_stack([X_sub, Z] + [X_sub * Z[:, [j]] for j in range(Z.shape[1])])
        compare(f"FWLS : poids selon {' + '.join(names)}", cross_fit(X, y, fold), ref, y, fold, exp)
    # biais résiduels : moyenne (rétrécie) des résidus de l'user / du film sur les autres plis
    for label, key in [("user", "user"), ("film", "item")]:
        for lam in [5, 20, 50]:
            corr = np.zeros_like(y)
            for k in np.unique(fold):
                tr = fold != k
                ids = oof[key]
                s = np.bincount(ids[tr], e[tr], minlength=ids.max() + 1)
                c = np.bincount(ids[tr], minlength=ids.max() + 1)
                corr[fold == k] = (s / (c + lam))[ids[fold == k]]
            compare(f"[test biaise] biais residuel par {label} (lambda={lam})", np.clip(sub + corr, .5, 5),
                    ref, y, fold, exp)
    # recalibration non linéaire : correction moyenne du résidu par tranche de prédiction
    edges = np.quantile(sub, np.linspace(0, 1, 21))
    corr = np.zeros_like(y)
    for k in np.unique(fold):
        tr = fold != k
        b_tr, b_te = np.digitize(sub[tr], edges[1:-1]), np.digitize(sub[fold == k], edges[1:-1])
        means = np.array([e[tr][b_tr == b].mean() for b in range(20)])
        corr[fold == k] = means[b_te]
    compare("recalibration par tranche de prediction (20 tranches)", np.clip(sub + corr, .5, 5), ref, y, fold, exp)

    # stacking non linéaire : gradient boosting sur le résidu du stack linéaire
    from sklearn.ensemble import HistGradientBoostingRegressor
    feats = np.column_stack([fm5, oof["knn_films"], chain_std, np.log1p(oof["n_i"]), np.log1p(oof["n_u"]),
                             oof["std_u"], oof["mean_u"], oof["half_u"], oof["mean_i"], oof["std_i"],
                             decades[oof["item"]], G[oof["item"]].astype(float)])
    for depth_name, params in [("leger", dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15)),
                               ("moyen", dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31))]:
        pred = np.empty_like(y)
        for k in np.unique(fold):
            tr, te = fold != k, fold == k
            w = lstsq(X_sub[tr], y[tr])
            base_tr, base_te = w[0] + X_sub[tr] @ w[1:], w[0] + X_sub[te] @ w[1:]
            gb = HistGradientBoostingRegressor(min_samples_leaf=200, l2_regularization=1.0, random_state=0,
                                               early_stopping=False, **params)
            gb.fit(feats[tr], y[tr] - base_tr)
            pred[te] = np.clip(base_te + gb.predict(feats[te]), 0.5, 5.0)
        compare(f"stack lineaire + gradient boosting sur le residu ({depth_name})", pred, ref, y, fold, exp)
        if depth_name == "moyen":
            gbdt_pred = pred
    # gain du gradient boosting par segment de popularité du film
    print("  gain du gradient boosting (moyen) par popularite du film :")
    for name, m in buckets(oof["n_i"], [0, 1, 2, 6, 21, 51, 10 ** 6], lambda lo, hi: f"[{lo},{hi})"):
        print(f"    {name:<14} {rmse(y[m] - gbdt_pred[m]) - rmse(y[m] - sub[m]):+.4f}")
    write_csv(f"{OUT_RES}/improvements.csv", exp)


if __name__ == "__main__":
    main()
