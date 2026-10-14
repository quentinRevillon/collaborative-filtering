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
data/                  # .npy du zip (non versionnés, cf. .gitignore)
generate.py            # point d'entrée de la plateforme (utilise src/)
src/
  data.py              # chargement des matrices + split validation
  metrics.py           # RMSE, accuracy (arrondi au 0.5)
  baselines.py         # GlobalMean, Biases (mu + b_u + b_i, régularisé ou non)
  mf.py                # factorisation R ≈ U I^T (+ mu, + biais) : descente de gradient et ALS
  run_baselines.py     # expérience : réglage des lambdas puis score train -> test
  run_mf_gd.py         # expérience : grille (rang, lambda) pour MF-GD -> results/
  run_als.py           # expérience : grille pour MF-ALS + comparaison GD vs ALS -> results/
  plot_convergence.py  # figure GD vs ALS (lit results/, écrit figures/)
results/               # CSV produits par les expériences
figures/               # figures pour les slides / le rapport
```

Les données sont des matrices `(610 users, 4980 films)`, NaN = note manquante.
Pour les obtenir : dézipper `datalab-a1.zip` dans `data/`.

Lancer les expériences depuis la racine : `python -m src.run_baselines`, `python -m src.run_mf_gd` (~10 min), `python -m src.run_als`, puis `python -m src.plot_convergence`
