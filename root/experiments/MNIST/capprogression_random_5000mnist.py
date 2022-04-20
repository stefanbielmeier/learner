import numpy as np
import matplotlib.pyplot as plt

from minmemorypoly import get_memorization_capacity
from estimate_cap import estimate_cap

num_memories = 5000
num_neurons = 784

random = np.random.randint(0,2,num_memories*num_neurons) 
randomarray = np.array(np.where(random == 0, -1, random), dtype=np.float64)
memories = np.reshape(randomarray,(num_memories, num_neurons))

print(memories.shape)

#x percentage of the dataset analyzed
dataset_share = np.array([0.02, 0.04, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) #0.04 is 2 images in each "class"

min_memorization_polydegrees = []

print("Num thresholds in 100 memories: ", estimate_cap(memories))

for share in dataset_share:
    
    selected = memories[0:int(share*num_memories), :]

    memorization_polydegree = get_memorization_capacity(selected)

    min_memorization_polydegrees.append(memorization_polydegree)


#plot
x_axis = dataset_share * 100
y2 = [2,4,10,20,30,40,50,60,70,80,90,100]

plt.plot(x_axis, min_memorization_polydegrees, label="capacity progression for % of dataset")
plt.plot(x_axis, y2, label="linear")

plt.xlabel("Percent of 100 images used as memories")
plt.ylabel("Capacity required to memorize (in terms of n)")

plt.legend(loc='best')
plt.show()


plt.plot(x_axis, min_memorization_polydegrees, label="capacity progression for % of dataset")
plt.plot(x_axis, y2, label="linear")

plt.xlabel("Percent of 100 images used as memories")
plt.ylabel("Capacity required to memorize (in terms of n)")

plt.legend(loc='best')

plt.ylim([0,20])
plt.show()
