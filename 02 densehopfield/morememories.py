import numpy as np
from actualcap import get_recall_qualities
import matplotlib.pyplot as plt

num_neurons = 16

for num_memories in range(1_000, 10_000,1_000):
    random = np.random.randint(0,2,num_memories*num_neurons) 
    randomarray = np.array(np.where(random == 0, -1, random), dtype=np.float64)

    data = np.reshape(randomarray, (num_memories,num_neurons)) 

    polydegrees = np.arange(1,20) #m
    recall_qualities_1000 = get_recall_qualities(data, polydegrees, num_neurons)

    plt.plot(polydegrees, recall_qualities_1000, label="num_memories: {}".format(num_memories))
    
plt.legend(loc="best")
plt.show()
