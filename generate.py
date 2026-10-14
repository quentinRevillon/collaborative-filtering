
import numpy as np
import os
from tqdm import tqdm, trange
import argparse

from src.bpmf import fit_bpmf
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
    

    # Bayesian matrix factorization with biases (see src/bpmf.py): rank 20, noise precision 2,
    # 100 warm-up Gibbs rounds, then the predictions of the next 300 rounds averaged
    predict = fit_bpmf(to_coo(table), table.shape, rank=20, n_samples=300, burn_in=100, alpha=2.0)
    table = complete(predict, table.shape)

    

    # Save the completed table 
    np.save("output.npy", table) ## DO NOT CHANGE THIS LINE


        
