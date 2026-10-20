# DataLab A1: Collaborative filtering (group 40, *La piscine*)

Completion of a sparse users × movies ratings matrix (MovieLens copy, 610 × 4980, ~1 % observed).
Metric: RMSE on held-out ratings (the platform also reports exact-rating accuracy and running time).

## Submitted method

`generate.py` fits a **Bayesian factorization machine** by Gibbs sampling (`src/bfm.py`, package
`myfm`): user and item ids, implicit feedback (items rated by the user, users who rated the item)
and the genres / release decade of `data/namesngenre2.npy`, rank 5. 118 Gibbs sweeps, the first
20 are discarded and the predictions of the other 98 are averaged; the noise precision is sampled.
Rank, features and number of sweeps were chosen by 5-fold cross-validation on `ratings_train`.

| Fitted on | Scored on | RMSE | Time |
|---|---|---|---|
| `ratings_train` | `ratings_test` | 0.8407 | ~5 s |
| `ratings_train` + `ratings_test` (platform input) | hidden eval set | (platform) | ~6 s |

Previous submission (BPMF, rank 20): 0.8646 on `ratings_test`. `data/namesngenre2.npy` is the
corrected genre file sent on 19/10 (the original `namesngenre.npy` was not aligned with the rating
columns: only 13 % of the columns had the right movie).

## Layout

```
generate.py              platform entry point (load/save lines unchanged)
requirements.txt         libraries installed by the platform (pip install -r requirements.txt);
                         torch is not listed: generate.py does not need it (see Usage)
slides.pdf               preliminary presentation (source: slides/slides.tex)
slides/                  slides.tex (main deck), slides_finales.tex/.pdf (deck with every method tested)
src/                     models, all with the API fit_xxx(train, shape, **hp) -> predict(u, i)
  data.py                loading, splits, metrics, matrix completion helper
  baselines.py           global mean, regularized biases mu + b_u + b_i
  mf.py, mf_plain.py     matrix factorization (ALS, gradient descent), with or without baseline
  knn.py, knn_course.py  item / user kNN (course formula and on bias residuals)
  bpmf.py                Bayesian matrix factorization (previous submission)
  bfm.py                 Bayesian factorization machines (submitted)
experiments/             reproducible experiment scripts (write results/ and figures/)
  run_course_methods.py  course methods: biases, matrix factorization, kNN (numbers of the slides)
  run_bayesian.py        BPMF and Bayesian factorization machines
  evaluate_output.py     score output.npy on a held-out file
results/, figures/       experiment outputs
docs/                    project procedure and documentation (LaTeX)
data/                    namesngenre2.npy (corrected genres, versioned), namesngenre.npy (original,
                         misaligned, kept for reference); ratings_*.npy are not versioned
```

## Usage

```bash
pip install -r requirements.txt
pip install torch                                          # only for the experiments below
python generate.py --name data/ratings_train.npy          # writes output.npy
python experiments/evaluate_output.py --output output.npy --test data/ratings_test.npy
```

`torch` is only used by the experiment code (`src/bpmf.py`, `src/mf.py`, `src/autorec.py`,
`src/gcmc.py`, `src/glocalk.py` and the `experiments/` scripts that call them); it is not in
`requirements.txt` so that the platform does not install it for `generate.py`.

## Protocol

Local: fit on `ratings_train`, hyperparameters tuned on a 10 % validation split of train (seed 0),
final score on `ratings_test`. Platform: `generate.py` receives train + test merged and is scored on a hidden eval set.

## Deliverables

`slides.pdf` and `report.pdf` at the repository root (A1 preliminary slides: 14 Oct 23:59; report: 21 Oct 23:59).
