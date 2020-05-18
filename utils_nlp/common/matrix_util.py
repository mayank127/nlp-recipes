import numpy as np

def length_normalize(matrix):
    norms = np.sqrt(np.sum(matrix**2, axis=1))
    norms[norms == 0] = 1
    return matrix / norms[:, np.newaxis]


def mean_center(matrix):
    avg = np.mean(matrix, axis=0)
    return matrix - avg