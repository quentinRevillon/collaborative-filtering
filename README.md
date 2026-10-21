# DataLab A1: Collaborative filtering (group 40, *La piscine*)

Completion of a sparse users × movies ratings matrix (MovieLens copy, 610 × 4980, ~1 % observed).
Metric: RMSE on held-out ratings (the platform also reports exact-rating accuracy and running time).

## Submitted method

`generate.py` fits the model of `src/final_stack.py`:

1. **Bayesian factorization machines** (`src/bfm.py`, package `myfm`): user and item ids, implicit
   feedback (items rated by the user, users who rated the item), genres / release decade of
   `data/namesngenre2.npy`; rank 20, 238 Gibbs sweeps (20 discarded), noise precision sampled,
   posterior means of 5 chains averaged. One FM is fitted on **user-normalized ratings**
   z = (r - m_u) / s_u (shrunk user mean and standard deviation, mapped back afterwards), one on
   raw ratings. The FM assumes the same noise for every user; on z, users whose ratings are very
   spread out (who concentrate the error) weigh less.
2. **Item kNN** on bias residuals (`src/knn.py`, `fit_item_knn_damped`).
3. **Linear stack** of the three predictions, corrected by **gradient boosting** (scikit-learn) on
   its residual, with features of the rating: predictions, disagreement between the chains,
   user / movie statistics, genres, decade.

Stack and boosting are learnt on out-of-fold predictions computed once by 5-fold cross-validation
on `ratings_train` + `ratings_test` (the platform input), versioned in `data/stack_train.npz`; they
are refitted in a few seconds by `generate.py`. If the input ratings differ from that file, the
boosting is skipped and a fixed linear stack is used.

| Model | 5-fold CV on train + test | `ratings_train` -> `ratings_test` | Platform |
|---|---|---|---|
| Bayesian FM rank 5, one chain (submission 2) | 0.8205 | 0.8407 | 0.812 |
| Bayesian FM rank 20, 5 chains + item kNN (submission 3) | 0.8098 | 0.8322 | |
| + user normalization (linear stack, fallback path) | 0.8046 | 0.8282 | |
| **+ gradient boosting (this submission)** | **0.8023** | - | |

The boosting cannot be scored on `ratings_test` (its training data contains the test ratings).
Running time on train + test: about 5.5 min. `data/namesngenre2.npy` is the corrected genre file
sent on 19/10 (the original `namesngenre.npy` was not aligned with the rating columns).

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
  knn.py, knn_course.py  item / user kNN (course formula and on bias residuals; damped variant submitted)
  bpmf.py                Bayesian matrix factorization (previous submission)
  bfm.py                 Bayesian factorization machines
  final_stack.py         submitted model: user-normalized FM + FM + kNN, stack + gradient boosting
experiments/             reproducible experiment scripts (write results/ and figures/)
  run_course_methods.py  course methods: biases, matrix factorization, kNN (numbers of the slides)
  run_bayesian.py        BPMF and Bayesian factorization machines
  evaluate_output.py     score output.npy on a held-out file
results/, figures/       experiment outputs
docs/                    project procedure and documentation (LaTeX)
data/                    namesngenre2.npy (corrected genres, versioned), stack_train.npz (out-of-fold
                         predictions for the stack), namesngenre.npy (original,
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
