# DataLab A1 — Collaborative filtering (group 40, *La piscine*)

Completion of a sparse users × movies ratings matrix (MovieLens copy, 610 × 4980, ~1 % observed).
Metric: RMSE on held-out ratings (the platform also reports exact-rating accuracy and running time).

## Layout

```
generate.py              platform entry point (do not change load/save lines)
src/                     models, all with the API fit_xxx(train, shape, **hp) -> predict(u, i)
  data.py                loading, splits, metrics, matrix completion helper
  baselines.py           global mean, regularized biases mu + b_u + b_i
experiments/             reproducible experiment scripts (write results/ and figures/)
  eda.py                 exploratory data analysis
  evaluate_output.py     score output.npy on a held-out file
figures/                 figures used in slides and report
docs/                    project procedure (LaTeX) and beginner course on the methods (docs/cours/)
data/                    local copy of the .npy files (not versioned)
```

## Usage

```bash
pip install -r requirements.txt
python generate.py --name data/ratings_train.npy          # writes output.npy
python experiments/evaluate_output.py --test data/ratings_test.npy
```

## Protocol

Local: fit on `ratings_train`, hyperparameters tuned on a 10 % validation split of train (seed 0),
final score on `ratings_test`. Platform: `generate.py` receives train + test merged and is scored on a hidden eval set.

## Deliverables

`slides.pdf` and `report.pdf` at the repository root (A1 preliminary slides: 14 Oct 23:59; report: 21 Oct 23:59).
