import numpy as np
# row stochastic adjacency matrix, of going from host to parasite
def average_resource_allocation(badj: np.array) -> np.array:
    A = badj / badj.sum(axis=1)[:, np.newaxis]
    At = badj.T / badj.T.sum(axis=1)[:, np.newaxis]
    weights = A @ At
    return (weights + weights.T) / 2