
import numpy as np
import os
from tqdm import tqdm, trange
import argparse

from src.baselines import fit_biases
from src.data import complete, to_coo


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a completed ratings table.')
    parser.add_argument("--name", type=str, default="ratings_train.npy",
                      help="Name of the npy of the ratings table to complete")

    args = parser.parse_args()



    # Open Ratings table
    print('Ratings loading...') 
    table = np.load(args.name) ## DO NOT CHANGE THIS LINE
    print('Ratings Loaded.')
    

    # Regularized biases mu + b_u + b_i (see src/baselines.py)
    predict = fit_biases(to_coo(table), table.shape, lam=2.0)
    table = complete(predict, table.shape)

    

    # Save the completed table 
    np.save("output.npy", table) ## DO NOT CHANGE THIS LINE


        
