
import numpy as np
import os
from tqdm import tqdm, trange
import argparse

from pathlib import Path

from src.bfm import fit_bfm
from src.data import complete, to_coo
from src.side_info import item_features

METADATA = Path(__file__).resolve().parent / "data" / "namesngenre.npy"  # versioned with the code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a completed ratings table.')
    parser.add_argument("--name", type=str, default="ratings_train.npy",
                      help="Name of the npy of the ratings table to complete")

    args = parser.parse_args()



    # Open Ratings table
    print('Ratings loading...') 
    table = np.load(args.name) ## DO NOT CHANGE THIS LINE
    print('Ratings Loaded.')
    

    # Bayesian factorization machine with implicit feedback and genres (see src/bfm.py)
    predict = fit_bfm(to_coo(table), table.shape, item_features(METADATA),
                      features=("iu", "ii", "side"), rank=10, n_iter=2000)
    table = complete(predict, table.shape)

    

    # Save the completed table 
    np.save("output.npy", table) ## DO NOT CHANGE THIS LINE


        
