import math

import numpy as np

T = np.array([[1,1,1,1,1],[-1,-1,1,-1,-1],[-1,-1,1,-1,-1],[-1,-1,1,-1,-1],[-1,-1,1,-1,-1]])
H = np.array([[1,-1,-1,-1,1],[1,-1,-1,-1,1],[1,1,1,1,1],[1,-1,-1,-1,1],[1,-1,-1,-1,1]])
E = np.array([[1,1,1,1,1], [1,-1,-1,-1,-1,], [1,1,1,1,1], [1,-1,-1,-1,-1], [1,1,1,1,1]])
X = np.array([[1,-1,-1,-1,1], [-1,1,-1,1,-1], [-1,-1,1,-1,-1], [-1,1,-1,1,-1], [1,-1,-1,-1,1]])

letters = np.stack([T,H,E, X], axis=0)
    
#flattens all dimensions except first dimension
letters = letters.reshape(letters.shape[0], -1)

random1 = np.array([[1,-1,1,-1,1], [-1,1,-1,1,-1], [1,-1,1,-1,1], [-1,1,-1,1,-1], [1,-1,1,-1,1]])
random2 = np.array([[-1,1,-1,1,-1], [1,-1,1,-1,1], [-1,1,-1,1,-1], [1,-1,1,-1,1], [-1,1,-1,1,-1]])

randomdata = np.stack([random1, random2, random1], axis=0)

minentropy1 = np.array([[1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1]])
minentropy2 = np.array([[-1,-1,-1,-1,-1], [-1,-1,-1,-1,-1], [-1,-1,-1,-1,-1], [-1,-1,-1,-1,-1], [-1,-1,-1,-1,-1]])

minentropy = np.stack([minentropy1, minentropy2, minentropy1])

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
print("min_entropy: ", estimate_n(minentropy))
print("letters: ", estimate_n(letters))