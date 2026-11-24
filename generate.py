"""Complete a ratings matrix with the submitted model and save it as output.npy.

The input is a users x movies .npy matrix with NaN for missing ratings. The model is the stack
of src/final_stack.py: Bayesian factorization machines (5 chains, rank 20) on user-normalized
and on raw ratings, item kNN, linear combination corrected by gradient boosting. Settings were
chosen by 5-fold cross-validation on ratings_train + ratings_test.

Usage (from the repository root):
    python generate.py --name data/ratings_train.npy     # writes output.npy

The load and save lines are fixed by the course evaluation platform.
"""
import argparse

import numpy as np

from src.data import complete, to_coo
from src.final_stack import fit_final_stack


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a completed ratings table.')
    parser.add_argument("--name", type=str, default="ratings_train.npy",
                        help="Name of the npy of the ratings table to complete")
    args = parser.parse_args()

    print('Ratings loading...')
    table = np.load(args.name)  ## DO NOT CHANGE THIS LINE
    print('Ratings Loaded.')

    predict = fit_final_stack(to_coo(table), table.shape)
    table = complete(predict, table.shape)

    np.save("output.npy", table)  ## DO NOT CHANGE THIS LINE
