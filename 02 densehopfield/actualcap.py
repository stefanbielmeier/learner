import math

import numpy as np
import matplotlib.pyplot as plt

from utils import plot_img
from densehopfield import HopfieldNetwork

#1 Memories
num_memories = 1_000
num_neurons = 16
num_examples = 1
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
int_constant_memories = np.reshape(constantarray, (num_memories, num_neurons))
float_constant_memories = np.array(int_constant_memories, dtype=np.float64)

print("int", int_memories)
print("float", float_memories)
print("const int", int_constant_memories)
print("const float", float_constant_memories)

#for plot
polydegrees = np.arange(1,50) #x

def get_recall_qualities(memories, polydegrees, num_neurons, network_max_cap = False):
    recall_qualities = []

    for n in polydegrees:
        #2 train dense hopfield network with 25 Neurons on desired memories
        network = HopfieldNetwork(num_neurons, n, max_cap = network_max_cap)
        network.learn(memories)

        #3 do prediction for 5 memories in dataset memory, see how many bits are the same (1 is 100%, 0 is 50% of bits are flipped => random) 
        num_experiments = 10
        randomidxs = np.random.randint(0,len(memories),num_experiments)

        avg_recall_quality = 0
        
        for idx in randomidxs:
            original = memories[idx].reshape(dims,dims)
            network.update(original.flatten())
            restored = network.get_state().reshape(dims,dims)
            num_equal_bits = np.sum(original.flatten() == restored.flatten())
            recall_quality = (num_equal_bits/num_neurons-0.5)*2
            avg_recall_quality = avg_recall_quality + recall_quality
        
        avg_recall_quality = avg_recall_quality/len(randomidxs)

        recall_qualities.append(avg_recall_quality)

    return recall_qualities

ints_recall_qualities = get_recall_qualities(int_memories, polydegrees, num_neurons)
float_recall_qualities = get_recall_qualities(float_memories, polydegrees, num_neurons)
constant_int_recall_qualities = get_recall_qualities(int_constant_memories, polydegrees, num_neurons)
constant_float_recall_qualities = get_recall_qualities(float_constant_memories, polydegrees, num_neurons)

plt.plot(polydegrees, ints_recall_qualities, float_recall_qualities)
plt.show()
plt.plot(polydegrees, constant_int_recall_qualities, constant_float_recall_qualities)
plt.show()