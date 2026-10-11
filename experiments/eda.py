"""Exploratory data analysis of the MovieLens-like collaborative filtering data.

Usage:
    python experiments/eda.py data

Prints key statistics to stdout and saves PNG figures in figures/eda/.
"""
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "figures" / "eda"
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
C_TRAIN, C_TEST = "#2a6fdb", "#e8833a"


def load(data_dir):
    """Load train/test rating matrices and movie metadata."""
    d = Path(data_dir)
    return np.load(d / "ratings_train.npy"), np.load(d / "ratings_test.npy"), np.load(d / "namesngenre.npy", allow_pickle=True)


def save(fig, name):
    """Save a figure as PNG in the output folder and close it."""
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {OUT / name}")


def basic_stats(R, T):
    """Print sparsity, overlap and rating moments; plot rating distributions."""
    mtr, mte = ~np.isnan(R), ~np.isnan(T)
    print(f"shape={R.shape}, n_train={mtr.sum()}, n_test={mte.sum()}, overlap={(mtr & mte).sum()}")
    print(f"density train={mtr.mean():.4%}, test={mte.mean():.4%}, union={(mtr | mte).mean():.4%}")
    vals = np.arange(0.5, 5.01, 0.5)
    ctr = np.array([(R[mtr] == v).sum() for v in vals]) / mtr.sum()
    cte = np.array([(T[mte] == v).sum() for v in vals]) / mte.sum()
    print(f"train mean={np.nanmean(R):.4f} std={np.nanstd(R):.4f} | test mean={np.nanmean(T):.4f} std={np.nanstd(T):.4f}")
    print("rating dist (train/test %):", {float(v): (round(100 * a, 2), round(100 * b, 2)) for v, a, b in zip(vals, ctr, cte)})
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(vals - 0.11, 100 * ctr, 0.22, label="train", color=C_TRAIN)
    ax.bar(vals + 0.11, 100 * cte, 0.22, label="test", color=C_TEST)
    ax.set(xlabel="Note", ylabel="% des notes", title="Distribution des notes", xticks=vals)
    ax.legend(frameon=False)
    save(fig, "01_rating_distribution.png")


def describe_counts(name, c):
    """Print summary statistics of a count vector."""
    q = np.percentile(c, [10, 25, 50, 75, 90, 99])
    print(f"{name}: min={c.min()} p10={q[0]:.0f} p25={q[1]:.0f} median={q[2]:.0f} mean={c.mean():.1f} p75={q[3]:.0f} p90={q[4]:.0f} p99={q[5]:.0f} max={c.max()}")


def count_stats(R, T):
    """Analyse ratings per user/movie and cold-start exposure of the test set."""
    mtr, mte = ~np.isnan(R), ~np.isnan(T)
    cu, ci = mtr.sum(1), mtr.sum(0)
    describe_counts("train ratings/user", cu)
    describe_counts("train ratings/movie", ci)
    describe_counts("test ratings/user", mte.sum(1))
    describe_counts("test ratings/movie", mte.sum(0))
    top = np.sort(ci)[::-1]
    print(f"top 10% movies hold {top[:len(top)//10].sum()/top.sum():.1%} of train ratings; top 1% hold {top[:len(top)//100].sum()/top.sum():.1%}")
    for k in [1, 5, 10]:
        print(f"users <{k} train: {(cu < k).sum()} | movies <{k} train: {(ci < k).sum()}")
    ui, ii = np.nonzero(mte)
    n = len(ui)
    for k in [1, 5, 10]:
        cu_t, ci_t = cu[ui] < k, ci[ii] < k
        print(f"test ratings with train count <{k}: user {cu_t.sum()} ({cu_t.mean():.2%}), movie {ci_t.sum()} ({ci_t.mean():.2%}), either {(cu_t | ci_t).sum()} ({(cu_t | ci_t).mean():.2%})")
    print(f"movies with 0 train AND 0 test: {((ci == 0) & (mte.sum(0) == 0)).sum()}")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, c, lab in [(axes[0], cu, "utilisateur"), (axes[1], ci, "film")]:
        ax.hist(c, bins=np.logspace(0, np.log10(c.max() + 1), 40) - 1 if lab == "film" else 40, color=C_TRAIN)
        ax.set(xlabel=f"# notes train par {lab}", ylabel="effectif", title=f"Notes par {lab} (train)")
    axes[1].set_xscale("symlog")
    axes[1].set_yscale("log")
    save(fig, "02_counts_hist.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(top) + 1) / len(top) * 100, np.cumsum(top) / top.sum() * 100, color=C_TRAIN, label="films")
    tu = np.sort(cu)[::-1]
    ax.plot(np.arange(1, len(tu) + 1) / len(tu) * 100, np.cumsum(tu) / tu.sum() * 100, color=C_TEST, label="utilisateurs")
    ax.set(xlabel="% des entités (triées par popularité)", ylabel="% cumulé des notes train", title="Longue traîne")
    ax.legend(frameon=False)
    save(fig, "03_long_tail.png")
    return cu, ci


def bias_stats(R, cu, ci):
    """Compute user and item biases (mean offsets) and plot their spread."""
    mu = np.nanmean(R)
    bu = np.nanmean(R - mu, axis=1)
    resid = R - mu - bu[:, None]
    with np.errstate(all="ignore"):
        bi = np.nanmean(resid, axis=0)
    bi_ok = bi[ci >= 5]
    print(f"global mean mu={mu:.4f}")
    print(f"user bias: std={np.nanstd(bu):.3f} p5={np.nanpercentile(bu,5):.3f} p95={np.nanpercentile(bu,95):.3f}")
    print(f"item bias (items>=5 ratings): std={bi_ok.std():.3f} p5={np.percentile(bi_ok,5):.3f} p95={np.percentile(bi_ok,95):.3f}; all items std={np.nanstd(bi):.3f}")
    ustd = np.nanstd(R, axis=1)
    print(f"per-user rating std: median={np.median(ustd):.3f}, users with std<0.3: {(ustd < 0.3).sum()}")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(bu, bins=40, color=C_TRAIN)
    axes[0].set(xlabel="b_u = moyenne utilisateur − μ", ylabel="effectif", title=f"Biais utilisateur (σ={np.nanstd(bu):.2f})")
    axes[1].hist(bi_ok, bins=40, color=C_TEST)
    axes[1].set(xlabel="b_i = moyenne résidu film", ylabel="effectif", title=f"Biais film, ≥5 notes (σ={bi_ok.std():.2f})")
    save(fig, "04_biases.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(ci[ci > 0], np.nanmean(R, axis=0)[ci > 0], s=5, alpha=0.4, color=C_TRAIN)
    ax.set_xscale("log")
    ax.set(xlabel="# notes train du film (log)", ylabel="note moyenne", title="Moyenne d'un film vs popularité")
    save(fig, "05_item_mean_vs_pop.png")


def parse_years(titles):
    """Extract the release year from titles of the form 'Name (YYYY)'; NaN if absent."""
    return np.array([float(m.group(1)) if (m := re.search(r"\((\d{4})\)\s*$", t.strip())) else np.nan for t in titles])


def metadata_stats(R, T, meta):
    """Genre counts, year extraction and mean rating by genre / decade."""
    genres = [g.split("|") for g in meta[:, 1]]
    cnt = Counter(g for gs in genres for g in gs)
    print("genres (n_movies):", dict(cnt.most_common()))
    print("genres per movie: mean", np.mean([len(g) for g in genres]))
    years = parse_years(meta[:, 0])
    print(f"years: parsed {np.isfinite(years).sum()}/{len(years)}, min={np.nanmin(years):.0f} median={np.nanmedian(years):.0f} max={np.nanmax(years):.0f}")
    print("titles without year:", list(meta[np.isnan(years), 0])[:10])
    mtr = ~np.isnan(R)
    ui, ii = np.nonzero(mtr)
    r = R[ui, ii]
    mu = r.mean()
    names = [g for g, _ in cnt.most_common()]
    rows = []
    for g in names:
        has = np.array([g in gs for gs in genres])
        sel = has[ii]
        rows.append((g, cnt[g], sel.sum(), r[sel].mean() if sel.any() else np.nan))
    print("genre | n_movies | n_train_ratings | mean_rating")
    for row in sorted(rows, key=lambda x: -np.nan_to_num(x[3])):
        print(f"  {row[0]:<20} {row[1]:5d} {row[2]:6d} {row[3]:.3f}")
    rows = [x for x in rows if x[2] >= 50]
    rows.sort(key=lambda x: x[3])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([x[0] for x in rows], [x[3] - mu for x in rows], color=[C_TRAIN if x[3] >= mu else C_TEST for x in rows])
    ax.axvline(0, color="grey", lw=0.8)
    ax.set(xlabel=f"note moyenne − μ ({mu:.2f})", title="Écart de note moyenne par genre (train)")
    save(fig, "06_genre_mean.png")
    fig, ax = plt.subplots(figsize=(7, 6))
    gc = sorted(cnt.items(), key=lambda x: x[1])
    ax.barh([g for g, _ in gc], [c for _, c in gc], color=C_TRAIN)
    ax.set(xlabel="# films", title="Nombre de films par genre")
    save(fig, "07_genre_counts.png")
    dec = (years[ii] // 10) * 10
    decs = np.unique(dec[np.isfinite(dec)])
    m = [r[dec == d].mean() for d in decs]
    n = [(dec == d).sum() for d in decs]
    print("decade | n_train_ratings | mean:", [(int(d), int(k), round(v, 3)) for d, k, v in zip(decs, n, m)])
    print("corr(year, rating) over train ratings:", np.corrcoef(years[ii][np.isfinite(years[ii])], r[np.isfinite(years[ii])])[0, 1])
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(decs, n, width=8, color="#c9d6ea", label="# notes")
    ax1.set(xlabel="Décennie de sortie", ylabel="# notes train")
    ax2 = ax1.twinx()
    ax2.plot(decs, m, "o-", color=C_TEST)
    ax2.set_ylabel("note moyenne", color=C_TEST)
    ax1.set_title("Note moyenne et volume par décennie (train)")
    save(fig, "08_decade_mean.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(years[np.isfinite(years)], bins=40, color=C_TRAIN)
    ax.set(xlabel="Année", ylabel="# films", title="Année de sortie des films")
    save(fig, "09_year_hist.png")


def svd_spectrum(R):
    """Singular values of the bias-removed, zero-filled train matrix; plot spectrum and energy."""
    mu = np.nanmean(R)
    bu = np.nanmean(R - mu, axis=1)
    with np.errstate(all="ignore"):
        bi = np.nan_to_num(np.nanmean(R - mu - bu[:, None], axis=0))
    X = np.nan_to_num(R - mu - bu[:, None] - bi[None, :])
    s = np.linalg.svd(X, compute_uv=False)
    X0 = np.nan_to_num(R - mu)
    s0 = np.linalg.svd(X0, compute_uv=False)
    e = np.cumsum(s ** 2) / np.sum(s ** 2)
    print("SV (bias-removed) top 15:", np.round(s[:15], 1))
    print("SV (mean-centered only) top 15:", np.round(s0[:15], 1))
    print("ratio s1/s2 (mean-centered):", round(s0[0] / s0[1], 2), "| bias-removed:", round(s[0] / s[1], 2))
    for k in [5, 10, 20, 50, 100, 200]:
        print(f"energy top-{k}: {e[k-1]:.2%}")
    # Marchenko-Pastur-like noise reference: SVD of the same mask filled with permuted residuals
    rng = np.random.default_rng(0)
    m = ~np.isnan(R)
    Xn = np.zeros_like(X)
    Xn[m] = rng.permutation(X[m])
    sn = np.linalg.svd(Xn, compute_uv=False)
    print("noise reference top 5 SV:", np.round(sn[:5], 1), "| #SV above noise max:", int((s > sn[0]).sum()))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    k = 100
    axes[0].plot(np.arange(1, k + 1), s0[:k], color="grey", label="centré (μ)")
    axes[0].plot(np.arange(1, k + 1), s[:k], color=C_TRAIN, label="μ + b_u + b_i retirés")
    axes[0].plot(np.arange(1, k + 1), sn[:k], "--", color=C_TEST, label="référence bruit (permutation)")
    axes[0].set(xlabel="rang", ylabel="valeur singulière", title="Spectre (100 premières)")
    axes[0].legend(frameon=False)
    axes[1].plot(np.arange(1, len(e) + 1), 100 * e, color=C_TRAIN)
    axes[1].set(xlabel="rang", ylabel="% énergie cumulée", title="Énergie cumulée (biais retirés)")
    save(fig, "10_svd_spectrum.png")


def bias_baseline(R, lam):
    """Return the prediction matrix mu + b_u + b_i with shrinkage lam (sequential estimation)."""
    m = ~np.isnan(R)
    mu = np.nanmean(R)
    bu = np.nansum(R - mu, axis=1) / (m.sum(1) + lam)
    bi = np.nansum(R - mu - bu[:, None], axis=0) / (m.sum(0) + lam)
    return mu + bu[:, None] + bi[None, :]


def baseline_rmse(R, T):
    """Print test RMSE of the global mean and of shrunk bias baselines."""
    m = ~np.isnan(T)
    print(f"RMSE global mean: {np.sqrt(np.mean((np.nanmean(R) - T[m]) ** 2)):.4f}")
    for lam in [0, 5, 10, 25]:
        print(f"RMSE mu+b_u+b_i (lambda={lam}): {np.sqrt(np.mean((bias_baseline(R, lam)[m] - T[m]) ** 2)):.4f}")


def main(data_dir):
    """Run the full EDA pipeline."""
    R, T, meta = load(data_dir)
    basic_stats(R, T)
    cu, ci = count_stats(R, T)
    bias_stats(R, cu, ci)
    baseline_rmse(R, T)
    metadata_stats(R, T, meta)
    svd_spectrum(R)


if __name__ == "__main__":
    main(sys.argv[1])
