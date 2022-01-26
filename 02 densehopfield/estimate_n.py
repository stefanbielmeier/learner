import math 
import matplotlib.pyplot as plt

import numpy as np

def estimate_n(table):
    """
    Takes: 2D numpy array representing an unlabeled dataset with rows as data points and columns as features.

    Returns: the approximated n for the dataset assuming 0% retrieval error

    Intuition: if the dataset contains less overlapping / complicated memories (orthogonal memories), n should be smaller because the hopfield net doesn't have to remember as many memories 
    but can extract features instead

    More thresholds => more complexity in the set => higher n required
    """
    row_sums = []
    column_sums = []

    for row in table:
        row_sums.append(np.sum(row))
    row_sums.sort(reverse=True)
    
    for column in np.transpose(table):
        column_sums.append(np.sum(column))

    column_sums.sort(reverse=True)

    #count thresholds in both
    row_thresholds = 1
    col_thresholds = 1
    
    curr_threshold = row_sums[0]
    for num in row_sums:
        if num != curr_threshold:
            curr_threshold = num
            row_thresholds = row_thresholds + 1
    
    curr_threshold = column_sums[0]
    for num in row_sums:
        if num != curr_threshold:
            curr_threshold = num
            col_thresholds = col_thresholds + 1

    return np.log(row_thresholds)/np.log(2), np.log(col_thresholds)/np.log(2)


row_thresholds = []
column_thresholds = []
datapoints = range(1_000,30_000,1_000)

for datapoint in datapoints:
    
    random = np.random.randint(0,2,datapoint) 
    randomarray = np.array(np.where(random == 0, -1, random), dtype=np.float64)

    
    arr = np.array([0.] * int(datapoint*0.8*3) + [1.] * int(datapoint*0.1*3) + [-1.] * int(datapoint*0.1*3)) # creates 1000 bits of noise for 48000 bits of data
    np.random.shuffle(arr) #shuffle noisy bitss
    d3_000 = np.tile(randomarray, 3)
    noisydata = np.where(arr == 0, d3_000, arr)

    noisyduplicateddata = np.reshape(np.concatenate((randomarray, noisydata)), (int(datapoint/25)*4,25))
        
    row, col = estimate_n(noisyduplicateddata)
    row_thresholds.append(row)
    column_thresholds.append(col)
    
plt.plot(datapoints, row_thresholds, label="row")
plt.plot(datapoints, column_thresholds, label="col")
plt.legend()
plt.show()
