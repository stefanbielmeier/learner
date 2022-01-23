from distutils.errors import DistutilsModuleError
import math

import numpy as np
import matplotlib.pyplot as plt

from utils import plot_img
from densehopfield import HopfieldNetwork

#1 create memories: 100 random binary patterns with size 25

num_memories = 15_000
num_neurons = 25
num_examples = 3
dims = int(math.sqrt(num_neurons))


#random
random = np.random.randint(0,2,num_memories*num_neurons)
randomarray = np.where(random == 0, -1, random)

#duplicate the dataset given #examples
random_int = []
for i in range(num_examples):
    random_int = np.concatenate((random_int, randomarray)) 
print(type(random_int))

#constant memories
constantarray = np.random.randint(1,2,num_memories*num_neurons)

memories = np.reshape(random_int,(num_examples*num_memories,num_neurons))
print(memories.shape)

#for plot
polydegrees = np.arange(1,50) #x
recall_qualities = [] #y

for n in polydegrees:
    #2 train dense hopfield network with 25 Neurons on desired memories
    network = HopfieldNetwork(num_neurons, n, max_cap = False)
    network.learn(memories)

    #3 do prediction for random one memory, see how many bits are the same (1 is 100%, 0 is 50% of bits are flipped => random) 
    randomidx = np.random.randint(0,len(memories),1)[0]

    original = memories[randomidx].reshape(dims,dims)

    network.update(original.flatten())

    restored = network.get_state().reshape(dims,dims)
    
    num_equal_bits = np.sum(original.flatten() == restored.flatten())
    
    recall_quality = (num_equal_bits/num_neurons-0.5)*2

    recall_qualities.append(recall_quality)

plt.plot(polydegrees, recall_qualities)
plt.show()