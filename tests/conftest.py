import numpy as np


def complex_random_sample(size):
    return np.random.random_sample(size) + (1j * np.random.random_sample(size))
