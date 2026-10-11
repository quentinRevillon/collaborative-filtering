"""Score a completed matrix (output.npy) on a held-out ratings file.

Usage (local protocol: fit on train, score on test):
    python generate.py --name data/ratings_train.npy
    python experiments/evaluate_output.py --output output.npy --test data/ratings_test.npy
"""
import argparse

import numpy as np


def score(output, test):
    """Return (rmse, accuracy after rounding to 0.5) of `output` on observed `test` entries."""
    mask = np.isfinite(test)
    pred, r = output[mask], test[mask]
    return float(np.sqrt(np.mean((pred - r) ** 2))), float(np.mean(np.round(pred * 2) / 2 == r))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="output.npy")
    parser.add_argument("--test", default="data/ratings_test.npy")
    args = parser.parse_args()
    rmse, acc = score(np.load(args.output), np.load(args.test))
    print(f"RMSE {rmse:.4f} | accuracy {acc:.4f}")
