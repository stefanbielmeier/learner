import numpy as np
import matplotlib.pyplot as plt

from estimate_cap import estimate_cap
from actualcap import get_recall_qualities

## Approach Feb 22nd:

## 1) have a 2-class resembling dataset
## 2) Estimate capacity in bits
## 3) Calculate the capacity in bits (0.24 bits / weight – MacKay) for a Hopfield network
## 4) Build a Hopfield Network with n = 2 and see what it does in terms of memorization (recall_quality)


dimensionality = 100

estimated = []
mackay = []
performance = []  

num_datapoint_range = range(10,100,5)

for num_datapoints in num_datapoint_range:
    #1) Create 2-class resembling dataset without labels.
    
    #create 2 uniuqe datapoints, one in each class
    random = np.random.randint(0, 2, num_datapoints*dimensionality)

    #make binary and reshape
    randomarray = np.array(np.where(random == 0, -1, random), dtype=np.float64)
    uniquepoints = np.reshape(randomarray, (num_datapoints,100))
        

    # 2) Estimate capacity in bits like with the supervised machine learner

    dataset_cap = estimate_cap(uniquepoints)
    estimated.append(dataset_cap)

    # 3) Calculate capacity of Hopfield Net in bits
    capacity_per_weight = 0.24
    num_weights = (dimensionality**2)/2

    network_capacity = capacity_per_weight * num_weights

    mackay.append(network_capacity)

    #4) The capacity of the network at predicted quality
    recallquality = get_recall_qualities(uniquepoints, polydegrees=[2], num_neurons=dimensionality)

    performance.append(recallquality[0])

#5) plot all

plt.plot(num_datapoint_range, estimated, label="estimate")
plt.plot(num_datapoint_range, mackay, label="mackay")
plt.plot(num_datapoint_range, performance, label="performance")

plt.legend(loc='best')
plt.show()

#6) do it for increasing # of datapoints to see when network starts failing

#TODO 
#CHANGE AXIS