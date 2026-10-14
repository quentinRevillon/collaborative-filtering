
import numpy as np
import os
from tqdm import tqdm, trange
import argparse

from pathlib import Path

from src.data import complete, to_coo
from src.side_info import item_features
from src.stack import fit_stack

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
    

    # Stack of Bayesian FMs, item kNN and a GBDT residual correction (see src/stack.py)
    predict = fit_stack(to_coo(table), table.shape, item_features(METADATA))
    table = complete(predict, table.shape)

    

    # Save the completed table 
    np.save("output.npy", table) ## DO NOT CHANGE THIS LINE


        
