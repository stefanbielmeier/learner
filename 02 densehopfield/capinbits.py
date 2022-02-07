import numpy as np

from estimate_cap import estimate_cap

## Approach Feb 22nd:

## 1) have a 2-class resembling dataset
## 2) Estimate capacity in bits
## 3) Calculate the capacity in bits (0.24 bits / weight – MacKay) for a Hopfield network
## 4) Build a Hopfield Network with n = 2 and see what it does in terms of recall_qualities


#1) Create 2-class resembling dataset without labels.

#create 2 uniuqe datapoints, one in each class
num_datapoints = 80
dimensionality = 100
random = np.random.randint(0, 2, num_datapoints*dimensionality)

#make binary and reshape
randomarray = np.array(np.where(random == 0, -1, random), dtype=np.float64)
uniquepoints = np.reshape(randomarray, (num_datapoints,100))
    

# 2) Estimate capacity in bits like with the supervised machine learner

dataset_cap = estimate_cap(uniquepoints)
print(dataset_cap)

# 3) Calculate capacity of Hopfield Net in bits
capacity_per_weight = 0.24
num_weights = (dimensionality**2)/2

network_capacity = capacity_per_weight * num_weights

print(network_capacity)

#4) Compare the two results...
