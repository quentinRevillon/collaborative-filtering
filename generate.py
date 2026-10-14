
import numpy as np
import os
from tqdm import tqdm, trange
import argparse

from src.baselines import Biases


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a completed ratings table.')
    parser.add_argument("--name", type=str, default="ratings_train.npy",
                      help="Name of the npy of the ratings table to complete")

    args = parser.parse_args()



    # Open Ratings table
    print('Ratings loading...') 
    table = np.load(args.name) ## DO NOT CHANGE THIS LINE
    print('Ratings Loaded.')
    

    # Any method you want
    # Baseline : biais user + film regularises (lambda regle sur validation, cf. src/run_baselines.py)
    table = Biases(lam_u=2, lam_i=2).fit(table).predict()

    

    # Save the completed table 
    np.save("output.npy", table) ## DO NOT CHANGE THIS LINE


        
