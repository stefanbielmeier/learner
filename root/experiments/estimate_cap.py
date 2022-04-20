import math
import matplotlib.pyplot as plt
import numpy as np


def estimate_cap(table):
    """
    Takes: 2D numpy array representing an unlabeled dataset with rows as data points and columns as features.

    Returns: the approximated capacity in bits required from a Hopfield network to memorize the whole set with close to 0% retrieval error

    Intuition: if the dataset contains less overlapping / complicated memories (orthogonal memories), n should be smaller because the hopfield net doesn't have to remember as many memories 
    but can extract features instead

    More thresholds => more complexity in the set => higher n required
    """
    row_sums = []

    for row in table:
        row_sums.append(np.sum(row))
    row_sums.sort(reverse=True)

    #count thresholds in both
    row_thresholds = 1

    curr_threshold = row_sums[0]
    for num in row_sums:
        #what if we vary the threshold equality to an inequalty?
        #if curr_threshold
        if num != curr_threshold:
            curr_threshold = num
            row_thresholds = row_thresholds + 1

    return math.log(row_thresholds)/math.log(2)*table.shape[1]


def main():
    #driver code
    print("it works by calling it from other programs!")

if __name__ == "__main__":
    main()