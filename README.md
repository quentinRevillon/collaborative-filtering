# DataLab A1: Collaborative filtering (group 40, *La piscine*)

Completion of a sparse users × movies ratings matrix (MovieLens copy, 610 × 4980, ~1 % observed).
Metric: RMSE on held-out ratings (the platform also reports exact-rating accuracy and running time).

## Submitted method

`generate.py` fits a **Bayesian matrix factorization with biases (BPMF)** by Gibbs sampling
(`src/bpmf.py`): rank 20, noise precision 2, 100 warm-up rounds, then the predictions of the
next 300 rounds are averaged. It is the most advanced method presented in `slides.pdf`.

| Fitted on | Scored on | RMSE | Time (4 CPU threads) |
|---|---|---|---|
| `ratings_train` | `ratings_test` | 0.8646 | ~1 min |
| `ratings_train` + `ratings_test` (platform input) | hidden eval set | (platform) | ~1.5 min |

## Layout

```
generate.py              platform entry point (load/save lines unchanged)
requirements.txt         libraries installed by the platform (pip install -r requirements.txt)
slides.pdf               preliminary presentation (source: slides/slides.tex)
slides/                  slides.tex (main deck), slides_finales.tex/.pdf (deck with every method tested)
src/                     models, all with the API fit_xxx(train, shape, **hp) -> predict(u, i)
  data.py                loading, splits, metrics, matrix completion helper
  baselines.py           global mean, regularized biases mu + b_u + b_i
  mf.py, mf_plain.py     matrix factorization (ALS, gradient descent), with or without baseline
  knn.py, knn_course.py  item / user kNN (course formula and on bias residuals)
  bpmf.py                Bayesian matrix factorization (submitted)
experiments/             reproducible experiment scripts (write results/ and figures/)
  run_course_methods.py  course methods: biases, matrix factorization, kNN (numbers of the slides)
  run_bayesian.py        BPMF and Bayesian factorization machines
  evaluate_output.py     score output.npy on a held-out file
results/, figures/       experiment outputs
docs/                    project procedure and documentation (LaTeX)
data/                    namesngenre.npy (versioned); ratings_*.npy are not versioned
```

## Usage

```bash
pip install -r requirements.txt
python generate.py --name data/ratings_train.npy          # writes output.npy
python experiments/evaluate_output.py --output output.npy --test data/ratings_test.npy
```

## Protocol

Local: fit on `ratings_train`, hyperparameters tuned on a 10 % validation split of train (seed 0),
final score on `ratings_test`. Platform: `generate.py` receives train + test merged and is scored on a hidden eval set.

## Deliverables

`slides.pdf` and `report.pdf` at the repository root (A1 preliminary slides: 14 Oct 23:59; report: 21 Oct 23:59).
