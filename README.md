# collaborative-filtering

Movie rating prediction on a sparse 610 x 4980 users x movies matrix, with Bayesian factorization machines stacked with an item kNN. Data Science Lab assignment, Master IASD (PSL), October 2026. The submission ranked 3rd out of 30 teams.

<img height="450" src="figures/final_rmse_by_segment.png" />

*Out-of-fold RMSE by user and movie segment: final model (CV 0.8023) against submission 3 (CV 0.8098).*

## Results

5-fold cross-validation RMSE on the 63,196 ratings of `ratings_train` + `ratings_test`, same folds for every model.

| Model | CV RMSE |
|---|---|
| Regularized biases (baseline) | 0.8642 |
| Item kNN (best course method) | 0.8523 |
| BPMF | 0.8479 |
| Bayesian FM, genres, implicit feedback, 5 chains | 0.8117 |
| + user normalization | 0.8049 |
| Stack with item kNN and gradient boosting (final) | 0.8023 |

Platform score on hidden ratings: 0.801 for the 5-chain Bayesian FM stacked with the item kNN. Details in the [report](docs/report.pdf) and [slides](docs/slides.pdf).

## Quick start

The course data (`ratings_train.npy`, `ratings_test.npy`, `namesngenre2.npy`) is not included. Place it in `data/`, then:

```bash
pip install -r requirements.txt
python generate.py --name data/ratings_train.npy
python experiments/evaluate_output.py --output output.npy --test data/ratings_test.npy
```

`generate.py` writes `output.npy`, the completed matrix (about 5.5 min on CPU).

## Method

- Bayesian factorization machine ([myFM](https://github.com/tohtsky/myFM)) fitted by Gibbs sampling, with user and movie one-hots, implicit feedback, genres and release decade. Rank 20, posterior means of 5 chains averaged.
- A second FM fitted on per-user normalized ratings, with user mean and standard deviation shrunk towards the global ones.
- Item kNN on bias residuals with shrunk Pearson similarity.
- Least-squares stack of the three predictions, corrected by gradient-boosted trees on out-of-fold predictions.
- Hyperparameters chosen by cross-validation on train + test.

## Structure

```
generate.py    platform entry point
src/           models (Bayesian FM, BPMF, MF, kNN, others tried)
experiments/   one script per experiment
results/       CSV results
figures/       plots
docs/          report, final slides, preliminary slides (PDF)
slides/        LaTeX sources of the slides
```

## References

1. R. Salakhutdinov et al. [Bayesian probabilistic matrix factorization using Markov chain Monte Carlo](https://doi.org/10.1145/1390156.1390267). ICML, 2008.
2. S. Rendle. [Factorization machines](https://doi.org/10.1109/ICDM.2010.127). ICDM, 2010.
3. C. Freudenthaler et al. [Bayesian factorization machines](https://www.ismll.uni-hildesheim.de/pub/pdfs/FreudenthalerRendle_BayesianFactorizationMachines.pdf). NIPS Workshop on Sparse Representation and Low-rank Approximation, 2011.

## Team

Quentin Revillon and Vincent Ami (group 40). Course by Alexandre Vérine, Université Paris Dauphine - PSL.
