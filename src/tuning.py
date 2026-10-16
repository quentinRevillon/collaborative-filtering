"""Réglage des hyperparamètres par validation croisée K plis sur le train.

Le test n'intervient jamais ici : il est réservé à src.evaluation.

Deux stratégies, qui renvoient le même format (lignes triées de la meilleure à la pire) :
    grid_search   : toutes les combinaisons d'une grille (1-2 hyperparamètres, courbes lisibles)
    optuna_search : recherche bayésienne TPE (3 hyperparamètres ou plus, plages continues)

Usage :
    folds = kfold_observed(R_train, n_folds=5)
    rows = grid_search(Biases, {"lam_u": [1, 2, 3], "lam_i": [1, 2, 3]}, folds)
    model = Biases(**best_params(rows[0]))
"""
import itertools

import numpy as np

from src.metrics import rmse

SCORE_COLUMNS = ("cv_rmse", "cv_std")


def cv_curves(model_cls, params, folds):
    """RMSE de validation de model_cls(**params) sur chaque pli : tableau (n_folds, n_epochs),
    avec n_epochs = 1 pour un modèle non itératif."""
    curves = []
    for R_fit, R_val in folds:
        model = model_cls(**params)
        if model.iterative:
            curves.append(model.fit(R_fit, R_val).history["val_rmse"])
        else:
            curves.append([rmse(R_val, model.fit(R_fit).predict())])
    return np.array(curves)


def cv_score(model_cls, params, folds):
    """Score CV d'une config : RMSE moyenne sur les plis et son écart-type entre plis.

    Pour un modèle itératif, on moyenne les courbes des plis époque par époque, et on
    garde l'époque où la courbe moyenne est minimale : n_epochs est remplacé par
    ce nombre (early stopping), prêt pour le réentraînement final.
    """
    return _summarize(model_cls, params, cv_curves(model_cls, params, folds))


def grid_search(model_cls, grid, folds, fixed=None, verbose=True):
    """Évalue par CV toutes les combinaisons de `grid` ({param: [valeurs]}), les
    paramètres `fixed` étant ajoutés tels quels. Renvoie les lignes triées de la
    meilleure à la pire (cv_rmse croissant)."""
    grid = grid if grid is not None else model_cls.param_grid()
    names = list(grid)
    rows = []
    for values in itertools.product(*grid.values()):
        row = cv_score(model_cls, {**(fixed or {}), **dict(zip(names, values))}, folds)
        rows.append(row)
        if verbose:
            _print_row(model_cls, dict(zip(names, values)), row)
    return sorted(rows, key=lambda r: r["cv_rmse"])


def optuna_search(model_cls, space, folds, fixed=None, n_trials=50, seed=0,
                  start_from=None, verbose=True):
    """Recherche bayésienne (Optuna, échantillonneur TPE) avec la même CV que grid_search.

    space : {param: spec}, spec parmi
        ("log", lo, hi)      réel, échelle log (régularisations, learning rate)
        ("float", lo, hi)    réel, échelle linéaire
        ("int", lo, hi)      entier ; ("int_log", lo, hi) entier en échelle log (rang)
        ("cat", [valeurs])   choix dans une liste
    start_from : configs (dicts) évaluées en premier, par ex. le meilleur point de la
    grille : la recherche ne peut alors pas faire moins bien que lui.

    Élagage (pruning) : après chaque pli, la RMSE moyenne des plis déjà faits est comparée
    à celle des autres essais au même stade ; un essai moins bon que la médiane est arrêté.

    Renvoie (rows, study) : rows au même format que grid_search (essais terminés,
    triés), study pour les analyses Optuna (importance des hyperparamètres...).
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1),
    )
    for config in start_from or []:
        study.enqueue_trial(config)

    def objective(trial):
        searched = {name: _suggest(trial, name, spec) for name, spec in space.items()}
        params = {**(fixed or {}), **searched}
        curves = []
        for f, fold in enumerate(folds):
            curves.append(cv_curves(model_cls, params, [fold])[0])
            trial.report(float(np.mean(curves, axis=0).min()), step=f)
            if trial.should_prune():
                if verbose:
                    print(f"    essai {trial.number:>3}  {_fmt(searched)}  elague apres {f + 1} pli(s)",
                          flush=True)
                raise optuna.TrialPruned()
        row = _summarize(model_cls, params, np.array(curves))
        trial.set_user_attr("row", row)
        if verbose:
            _print_row(model_cls, searched, row, prefix=f"essai {trial.number:>3}  ")
        return row["cv_rmse"]

    study.optimize(objective, n_trials=n_trials)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    rows = sorted((t.user_attrs["row"] for t in done), key=lambda r: r["cv_rmse"])
    return rows, study


def best_params(row):
    """Hyperparamètres d'une ligne de résultats (sans les colonnes de score). Les cases
    vides d'un CSV qui mélange plusieurs variantes (None) sont ignorées : le modèle
    garde alors sa valeur par défaut."""
    return {k: v for k, v in row.items()
            if k not in SCORE_COLUMNS and k != "variant" and v is not None}


def _summarize(model_cls, params, curves):
    mean = curves.mean(axis=0)
    best = int(np.argmin(mean))
    row = dict(params)
    if model_cls.iterative:
        row["n_epochs"] = best + 1
    row["cv_rmse"] = round(float(mean[best]), 4)
    row["cv_std"] = round(float(curves[:, best].std()), 4)
    return row


def _suggest(trial, name, spec):
    kind, *args = spec
    if kind == "log":
        return trial.suggest_float(name, *args, log=True)
    if kind == "float":
        return trial.suggest_float(name, *args)
    if kind == "int":
        return trial.suggest_int(name, *args)
    if kind == "int_log":
        return trial.suggest_int(name, *args, log=True)
    if kind == "cat":
        return trial.suggest_categorical(name, args[0])
    raise ValueError(f"type d'espace inconnu : {kind}")


def _fmt(params):
    return "  ".join(f"{n}={v:.3g}" if isinstance(v, float) else f"{n}={v}" for n, v in params.items())


def _print_row(model_cls, searched, row, prefix=""):
    epochs = f"  ({row['n_epochs']} ep.)" if model_cls.iterative else ""
    print(f"    {prefix}{_fmt(searched)}  cv={row['cv_rmse']:.4f} +/- {row['cv_std']:.4f}{epochs}",
          flush=True)
