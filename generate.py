
import numpy as np
import os
from tqdm import tqdm, trange
import argparse

from pathlib import Path

from src.bfm import fit_bfm
from src.data import complete, to_coo
from src.side_info import item_features

GENRES = Path(__file__).resolve().parent / "data" / "namesngenre2.npy"   # corrected genre file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a completed ratings table.')
    parser.add_argument("--name", type=str, default="ratings_train.npy",
                      help="Name of the npy of the ratings table to complete")

    args = parser.parse_args()



    # Open Ratings table
    print('Ratings loading...') 
    table = np.load(args.name) ## DO NOT CHANGE THIS LINE
    print('Ratings Loaded.')
    

    # Bayesian factorization machine (see src/bfm.py): user/item ids + implicit feedback
    # (items rated by the user, users who rated the item) + genres/decade, rank 5;
    # 118 Gibbs sweeps, the first 20 discarded, the predictions of the other 98 averaged.
    # Rank, features and number of sweeps chosen by 5-fold cross-validation on ratings_train.
    predict = fit_bfm(to_coo(table), table.shape, item_features(GENRES),
                      features=("iu", "ii", "side"), rank=5, n_iter=118, burn_in=20)
    table = complete(predict, table.shape)

    

    # Save the completed table 
    np.save("output.npy", table) ## DO NOT CHANGE THIS LINE


        
