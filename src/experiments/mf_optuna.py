"""Régularisations séparées lam_u (facteurs users), lam_i (facteurs films) et lam_bias,
réglées avec Optuna, pour la MF avec biais : r̂_ui = mu + b_u + b_i + <u_u, i_i>.

Protocole :
  1. recherche Optuna (TPE + élagage) par CV 5 plis sur le train, avec l'ALS (même
     minimum que la GD, ~5x plus rapide) ; le premier essai est le meilleur point de la
     grille ALS (src/experiments/als.py), ou à défaut de la grille GD (mf_gd.py) :
     lam_u = lam_i, lam_bias = 2, pour mesurer ce qu'apporte la séparation ;
  2. importance des hyperparamètres (fANOVA d'Optuna) ;
  3. la meilleure config est réentraînée sur tout le train et évaluée sur le test, avec
     l'ALS et la MF-GD de la grille, et la baseline biais en référence. Pour la GD, le
     nombre d'époques des hyperparamètres Optuna est choisi par la même CV.
Sorties : results/mf_optuna_trials.csv, results/mf_optuna_importance.csv, results/mf_optuna_test.csv.

Lancer depuis la racine du repo :  python -m src.experiments.mf_optuna [--folds 5] [--trials 25]
"""
import argparse

from src.data import kfold_observed, load_train_test
from src.evaluation import best_row, evaluate_on_test, write_csv
from src.models import Biases, MatrixFactorizationALS, MatrixFactorizationGD
from src.tuning import best_params, cv_score, optuna_search

SPACE = {
    "rank": ("int_log", 2, 64),
    "lam_u": ("log", 0.1, 100.0),
    "lam_i": ("log", 0.1, 100.0),
    "lam_bias": ("log", 0.1, 100.0),
}
FIXED = dict(bias="full", n_epochs=25)
GD = dict(n_epochs=800, lr=3e-4, momentum=0.9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trials", type=int, default=25)
    args = parser.parse_args()

    R_train, R_test = load_train_test()
    folds = kfold_observed(R_train, n_folds=args.folds)

    # Point de départ : le meilleur de la grille ALS, sinon de la grille GD
    # (une seule régularisation des facteurs, lam_bias = 2)
    grid_row = best_row("als_grid.csv", "full")
    gd_grid = best_row("mf_gd_grid.csv", "full")
    start_row = grid_row or gd_grid
    start = (dict(rank=start_row["rank"], lam_u=float(start_row["lam"]),
                  lam_i=float(start_row["lam"]), lam_bias=2.0) if start_row
             else dict(rank=50, lam_u=10.0, lam_i=10.0, lam_bias=2.0))

    print(f"Optuna ALS full, {args.trials} essais, CV {args.folds} plis "
          f"(essai 0 = meilleur point de la grille {'ALS' if grid_row else 'GD'}) :")
    rows, study = optuna_search(MatrixFactorizationALS, SPACE, folds, fixed=FIXED,
                                n_trials=args.trials, start_from=[start])
    write_csv("mf_optuna_trials.csv", [
        dict(trial=t.number, state=t.state.name, **t.params,
             cv_rmse=t.user_attrs.get("row", {}).get("cv_rmse"),
             cv_std=t.user_attrs.get("row", {}).get("cv_std"),
             n_epochs=t.user_attrs.get("row", {}).get("n_epochs"))
        for t in study.trials])

    import optuna
    importance = optuna.importance.get_param_importances(study)
    print("\nImportance des hyperparametres (part de la variance du score expliquee) :")
    for name, value in importance.items():
        print(f"  {name:<9} {value:.2f}")
    write_csv("mf_optuna_importance.csv", [dict(param=n, importance=round(v, 3))
                                           for n, v in importance.items()])

    best = best_params(rows[0])
    print(f"\nMeilleur : {best}  cv={rows[0]['cv_rmse']} +/- {rows[0]['cv_std']}")

    gd_params = {k: v for k, v in best.items() if k != "n_epochs"} | GD
    print("Nombre d'epoques de la GD pour ces hyperparametres (CV) :")
    gd_row = cv_score(MatrixFactorizationGD, gd_params, folds)
    print(f"  {gd_row['n_epochs']} ep.  cv={gd_row['cv_rmse']} +/- {gd_row['cv_std']}")

    models = {}
    baseline = best_row("baselines_grid.csv", "both")
    models["Baseline biais regularises"] = (Biases(**best_params(baseline)) if baseline
                                            else Biases(lam_u=5, lam_i=3))
    if grid_row:
        p = best_params(grid_row)
        models[f"MF-ALS grille (k={p['rank']}, lam={p['lam']})"] = MatrixFactorizationALS(**p)
    if gd_grid:
        p = best_params(gd_grid)
        models[f"MF-GD grille (k={p['rank']}, lam={p['lam']})"] = MatrixFactorizationGD(**p)
    tag = (f"k={best['rank']}, lam_u={best['lam_u']:.3g}, lam_i={best['lam_i']:.3g}, "
           f"lam_b={best['lam_bias']:.3g}")
    models[f"MF-ALS Optuna ({tag})"] = MatrixFactorizationALS(**best)
    models[f"MF-GD Optuna ({gd_row['n_epochs']} ep.)"] = MatrixFactorizationGD(**best_params(gd_row))
    evaluate_on_test(models, R_train, R_test, csv_name="mf_optuna_test.csv")


if __name__ == "__main__":
    main()
