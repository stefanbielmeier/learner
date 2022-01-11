import math

import numpy as np

#create some random data
random = np.random.randint(0,2,100)
randomarray = np.where(random == 0, -1, random)
randomdata = np.reshape(randomarray,(4,25))


#bunch of ones
constant_data = np.reshape(np.random.randint(1,2,100), (4,25))


def estimate_n(table):
    """
    Takes: 2D numpy array representing an unlabeled dataset with rows as data points and columns as features.

    Returns: the approximated n for the dataset assuming 0% retrieval error

    Intuition: if the dataset contains less overlapping / complicated memories (orthogonal memories), n should be smaller because the hopfield net doesn't have to remember as many memories 
    but can extract features instead

    More thresholds => more complexity in the set => higher n required
    """
    sums = []
    curr_threshold = 0
    num_thresholds = 1 #at least one!

    for row in table:
        sums.append(np.sum(row))
    
    sums.sort(reverse=True)
    curr_threshold = sums[0]
    print(sums)

    for sum in sums[1:]:
        if sum != curr_threshold:
            num_thresholds = num_thresholds + 1
            curr_threshold = sum

    return {
        "thresholds": num_thresholds,
        "average" : math.floor(np.average(sums)),
    }

print("random: ", estimate_n(randomdata))
print("contant: ", estimate_n(constant_data))