from distutils.errors import DistutilsModuleError
import math

import numpy as np
import matplotlib.pyplot as plt

from utils import plot_img
from densehopfield import HopfieldNetwork

#1 Memories
num_memories = 200
num_neurons = 16
num_examples = 2
dims = int(math.sqrt(num_neurons))

#2 create random memories
random = np.random.randint(0,2,num_memories*num_neurons)
randomarray = np.where(random == 0, -1, random)

#extend the random data #num_examples of times
random_ints = np.tile(randomarray, num_examples)

#store as float, as well
random_floats = np.array(random_ints, dtype=np.float64)

#constant memories
constantarray = np.random.randint(1,2,num_memories*num_neurons)

int_memories = np.reshape(random_ints,(num_examples*num_memories,num_neurons))
float_memories = np.reshape(random_floats,(num_examples*num_memories,num_neurons))
constant_memories = np.reshape(constantarray, (num_memories, num_neurons))

print("float", int_memories)
print("int", float_memories)
print("const", constant_memories)

#for plot
polydegrees = np.arange(1,50) #x
ints_recall_qualities = [] 
float_recall_qualities = []

for n in polydegrees:
    #2 train dense hopfield network with 25 Neurons on desired memories
    network = HopfieldNetwork(num_neurons, n, max_cap = False)
    network.learn(int_memories)

    float_net = HopfieldNetwork(num_neurons, n, max_cap = False)
    float_net.learn(float_memories)

    #3 do prediction for random one memory, see how many bits are the same (1 is 100%, 0 is 50% of bits are flipped => random) 
    randomidx = np.random.randint(0,len(int_memories),1)[0]

    original = int_memories[randomidx].reshape(dims,dims)

    network.update(original.flatten())

    restored = network.get_state().reshape(dims,dims)
    
    num_equal_bits = np.sum(original.flatten() == restored.flatten())
    
    recall_quality = (num_equal_bits/num_neurons-0.5)*2

    ints_recall_qualities.append(recall_quality)

    #forfloats
    randomidx = np.random.randint(0,len(float_memories),1)[0]

    original = float_memories[randomidx].reshape(dims,dims)

    float_net.update(original.flatten())

    restored = float_net.get_state().reshape(dims,dims)
    
    num_equal_bits = np.sum(original.flatten() == restored.flatten())
    
    recall_quality = (num_equal_bits/num_neurons-0.5)*2

    float_recall_qualities.append(recall_quality)

plt.plot(polydegrees, ints_recall_qualities, float_recall_qualities)
plt.show()