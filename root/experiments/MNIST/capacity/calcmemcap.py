import numpy as np
import matplotlib.pyplot as plt

from root.experiments.MNIST.digits.subsets import get_first_fifty_images
from root.experiments.MNIST.capacity.definitions import DATASET_SHARE
from root.experiments.MNIST.minmemorypoly import get_memorization_capacity

def calc_memorization_capacities(first_set, second_set = None, has_labels = True):

    min_memorization_polydegrees = []
    if has_labels:
        first_set = first_set[:, :-1]
        if second_set is not None:
            second_set = second_set[:, :-1]

    for share in DATASET_SHARE:
        zero_subset = first_set[0:int(share*50), :]
        one_subset = second_set[0:int(share*50), :]

        partial_set = np.concatenate((zero_subset, one_subset))

        memorization_polydegree = get_memorization_capacity(partial_set)

        min_memorization_polydegrees.append(memorization_polydegree)

    return min_memorization_polydegrees

if __name__ == "__main__":
    zeros, ones, sixes, eights = get_first_fifty_images(inBinary = True)
    mem_caps = calc_memorization_capacities(sixes, eights, has_labels=True)
    print(mem_caps)