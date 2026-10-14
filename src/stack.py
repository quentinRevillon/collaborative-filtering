"""Final model: linear stack of the Bayesian FM, item kNN, a second FM and the GBDT-corrected FM.

    r_hat = w0 + w_fm FM + w_knn kNN + w_iu_ii FM(iu, ii) + w_gbdt (FM + GBDT correction)

The weights were learned by ridge regression on the validation split and selected there
(experiments/run_stack.py, results/stack_weights.csv); they are frozen here so that every
member is fitted only once on the input ratings (time budget of the platform).

Usage:
    from src.side_info import item_features
    from src.stack import fit_stack
    predict = fit_stack(train, shape, item_features("data/namesngenre.npy"))   # train: COO (u, i, r)
    r_hat = predict(u, i)
"""
from src.bfm import fit_bfm
from src.knn import fit_item_knn
from src.residual import fold_ids, oof_predictions
from src.residual_gbdt import fit_gbdt_correction

WEIGHTS = dict(fm=0.0129, knn=0.2466, fm_iu_ii=0.2717, fm_gbdt=0.5328)
INTERCEPT = -0.2463
FOLD_ITER = 500     # Gibbs iterations of the out-of-fold FMs that train the GBDT correction


def fit_gbdt(train, shape, F):
    """GBDT correction of the FM learned on its 5-fold out-of-fold residuals: correct(p, u, i)."""
    fold = fold_ids(len(train[2]))
    fit_fold = lambda d: fit_bfm(d, shape, F, n_iter=FOLD_ITER)
    return fit_gbdt_correction(train, shape, oof_predictions(train, fit_fold, fold), fold)


def fit_stack(train, shape, F):
    """Fit every member once on `train` and return predict(u, i) = frozen linear stack."""
    fm = fit_bfm(train, shape, F, n_iter=2000)
    fm_iu_ii = fit_bfm(train, shape, F, features=("iu", "ii"), n_iter=1000)
    knn = fit_item_knn(train, shape, k=40, shrink=100.0, beta=0.2, lam=2.0)
    correct = fit_gbdt(train, shape, F)

    def predict(u, i):
        p = fm(u, i)
        return (INTERCEPT + WEIGHTS["fm"] * p + WEIGHTS["knn"] * knn(u, i)
                + WEIGHTS["fm_iu_ii"] * fm_iu_ii(u, i) + WEIGHTS["fm_gbdt"] * (p + correct(p, u, i)))

    return predict
