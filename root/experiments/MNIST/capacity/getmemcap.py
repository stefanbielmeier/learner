import numpy as np
import matplotlib.pyplot as plt

from root.experiments.MNIST.digits.subsets import get_first_fifty_images
from root.experiments.MNIST.capacity.definitions import DATASET_SHARE
from root.experiments.MNIST.minmemorypoly import get_memorization_capacity

def get_memorization_capacities(first_set, second_set = None):

    min_memorization_polydegrees = []
    for share in DATASET_SHARE:
        
        zero_subset = first_set[0:int(share*50), :]
        one_subset = second_set[0:int(share*50), :]

        partial_set = np.concatenate((zero_subset, one_subset))

        memorization_polydegree = get_memorization_capacity(partial_set)

        min_memorization_polydegrees.append(memorization_polydegree)

    return min_memorization_polydegrees

if __name__ == "__main__":
    zeros, ones, sixes, eights = get_first_fifty_images(inBinary = True)
    mem_caps = get_memorization_capacities(sixes, eights)
    print(mem_caps)