
import numpy as np
import os
from tqdm import tqdm, trange
import argparse

from pathlib import Path

from src.bfm import fit_bfm_chains
from src.data import complete, to_coo
from src.knn import fit_item_knn_damped
from src.side_info import item_features

GENRES = Path(__file__).resolve().parent / "data" / "namesngenre2.npy"   # corrected genre file

# Stack weights, learnt by least squares on out-of-fold predictions (5-fold CV on train + test)
N_CHAINS = 5
W_INTERCEPT, W_FM, W_KNN = 0.0657, 0.8215, 0.1609


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a completed ratings table.')
    parser.add_argument("--name", type=str, default="ratings_train.npy",
                      help="Name of the npy of the ratings table to complete")

    args = parser.parse_args()



    # Open Ratings table
    print('Ratings loading...') 
    table = np.load(args.name) ## DO NOT CHANGE THIS LINE
    print('Ratings Loaded.')
    

    # Stack of two models (settings chosen by 5-fold cross-validation on train + test):
    # 1. Bayesian factorization machine (src/bfm.py): user/item ids + implicit feedback (items
    #    rated by the user, users who rated the item) + genres/decade, rank 20; 238 Gibbs sweeps,
    #    the first 20 discarded; posterior means of N_CHAINS chains (seeds 0..N_CHAINS-1) averaged;
    # 2. item kNN on bias residuals (src/knn.py, fit_item_knn_damped);
    # combined linearly with the weights above (complete() clips to [0.5, 5]).
    train = to_coo(table)
    fm = fit_bfm_chains(train, table.shape, item_features(GENRES), features=("iu", "ii", "side"),
                        rank=20, n_iter=238, seeds=tuple(range(N_CHAINS)), burn_in=20)
    knn = fit_item_knn_damped(train, table.shape, k=20, shrink=100.0, damp=0.5, lam_u=5.0, lam_i=3.0)
    predict = lambda u, i: W_INTERCEPT + W_FM * fm(u, i) + W_KNN * knn(u, i)
    table = complete(predict, table.shape)

    

    # Save the completed table 
    np.save("output.npy", table) ## DO NOT CHANGE THIS LINE


        
