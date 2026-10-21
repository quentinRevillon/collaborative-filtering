
import numpy as np
import os
from tqdm import tqdm, trange
import argparse

from src.data import complete, to_coo
from src.final_stack import fit_final_stack


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a completed ratings table.')
    parser.add_argument("--name", type=str, default="ratings_train.npy",
                      help="Name of the npy of the ratings table to complete")

    args = parser.parse_args()



    # Open Ratings table
    print('Ratings loading...') 
    table = np.load(args.name) ## DO NOT CHANGE THIS LINE
    print('Ratings Loaded.')
    

    # Stack (see src/final_stack.py): Bayesian FMs fitted on user-normalized ratings and on raw
    # ratings (5 chains each, rank 20), item kNN, linear combination corrected by gradient boosting.
    # Settings chosen by 5-fold cross-validation on ratings_train + ratings_test.
    predict = fit_final_stack(to_coo(table), table.shape)
    table = complete(predict, table.shape)

    

    # Save the completed table 
    np.save("output.npy", table) ## DO NOT CHANGE THIS LINE


        
