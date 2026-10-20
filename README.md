# DataLabAssignement1

## generate.py
Use the file *generate.py* to complete your ratings table. 
It takes in argument *--name* the name of the files you want to use and it saves the complete matrix as *output.npy*.
DO NOT CHANGE THE LINES TO LOAD AND SAVE THE TABLE. Between those to you are free to use any method for matrix completion. 
Example:
  > python3 generate.py --name ratings_train.npy

## requirements.txt
Among the good pratice of datascience, we encourage you to use conda or virtualenv to create python environment. 
To test your code on our platform, you are required to update the *requirements.txt*, with the different librairies you might use. 
When your code will be test, we will execute: 
  > pip install -r requirements.txt

## Organisation du code

```
data/                    # .npy du zip (non versionnés), sauf namesngenre2.npy (genres corrigés, versionné)
generate.py              # point d'entrée de la plateforme (utilise src.models)
src/
  data.py                # chargement des matrices, split validation, K-fold sur les notes observées
  metrics.py             # RMSE, accuracy (arrondi au 0.5)
  models/                # un fichier par famille de méthodes, interface fit(R, R_val) / predict()
    base.py              #   classe Model commune, bornes des notes
    baselines.py         #   GlobalMean, Biases (mu + b_u + b_i, régularisé ou non)
    mf.py                #   R ≈ U I^T (+ mu, + biais) : descente de gradient et ALS
    knn.py               #   kNN sur les résidus des biais (Pearson + shrink + damp)
    koren.py             #   modèle de voisinage de Koren 2008 (poids appris)
    bpmf.py              #   BPMF (Gibbs), alpha fixé ou tiré
    bfm.py               #   Bayesian Factorization Machine (myfm) : implicite + genres
  tuning.py              # validation croisée K plis : grid_search, optuna_search (early stopping inclus)
  evaluation.py          # protocole final : réentraîner sur train, scorer sur test, CSV
  experiments/           # une étude = un script court qui assemble les briques ci-dessus
    baselines.py
    mf_gd.py
    als.py
    mf_optuna.py         #   lam_u, lam_i, lam_bias séparés, réglés avec Optuna
    knn.py, koren.py     #   voisinages : kNN films/users, Koren
    bpmf.py              #   BPMF : variantes, rang
    bpmf_init.py         #   BPMF : effet de l'initialisation
    bpmf_alpha.py        #   BPMF : grille sur alpha, alpha tiré, config longue, test
    bfm.py               #   Bayesian FM : features, rang, courbe, test
  plots/
    convergence.py       # figure GD vs ALS (lit results/, écrit figures/)
    bpmf_alpha.py        # figure RMSE CV de BPMF selon alpha
results/                 # CSV et logs produits par les expériences
figures/                 # figures pour les slides / le rapport
```

Les données sont des matrices `(610 users, 4980 films)`, NaN = note manquante.
Pour les obtenir : dézipper `datalab-a1.zip` dans `data/`.

### Protocole

1. Hyperparamètres choisis par **validation croisée 5 plis** sur `ratings_train` (chaque pli : 80 % des
   notes pour apprendre, 20 % pour valider). Pour les modèles itératifs, le nombre d'époques est
   pris au minimum de la courbe de validation moyenne.
2. Le modèle retenu est réentraîné sur tout le train, puis scoré **une seule fois** sur `ratings_test`.

### Ajouter une méthode

1. Un fichier dans `src/models/` avec une classe qui hérite de `Model` (`fit(R, R_val=None)`,
   `predict()`, et `param_grid()` pour sa grille par défaut). Modèle itératif : `iterative = True`,
   paramètre `n_epochs`, et `self.history["val_rmse"]` rempli pendant `fit`.
2. L'ajouter à `MODELS` dans `src/models/__init__.py`.
3. Un script dans `src/experiments/` : `grid_search(...)` (1-2 hyperparamètres) ou
   `optuna_search(...)` (3 ou plus), puis `evaluate_on_test(...)`.

### Lancer

```
pip install -r requirements.txt
python -m src.experiments.baselines        # ~15 s
python -m src.experiments.mf_gd            # long (~30 min)
python -m src.experiments.als              # ~15 min, utilise les résultats des deux précédents
python -m src.experiments.mf_optuna        # ~30 min (25 essais), part du meilleur point de la grille
python -m src.experiments.knn              # ~1 min
python -m src.experiments.koren            # ~4 min
python -m src.experiments.bpmf             # ~10 min
python -m src.experiments.bpmf_alpha       # ~25 min (options --gibbs, --long, --test)
python -m src.experiments.bfm              # ~20 min (--skip-cv : seulement courbe + test)
python -m src.plots.convergence
python -m src.plots.bpmf_alpha
```
