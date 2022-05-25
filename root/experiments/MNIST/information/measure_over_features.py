import numpy as np
from root.experiments.MNIST.digits.subsets import get_first_fifty_images
from root.experiments.MNIST.digits.subsets import make_binary

from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

zeros, ones, sixes, eights = get_first_fifty_images(inBinary=False)

num_neurons = 784
num_memories = 50

uniform_random = np.random.randint(0, 2, num_memories)

def calculate_bits (dataset, uniform_random):
    bit_divs = []
    for column in dataset.T:
        bit_divs.append(jensenshannon(column, uniform_random, base=2))
    bit_divs = np.array(bit_divs, dtype=np.float64)
    bit_divs = np.nan_to_num(bit_divs)
    return np.sum(bit_divs)

print(calculate_bits(zeros, uniform_random))
print(calculate_bits(ones, uniform_random))
print(calculate_bits(sixes, uniform_random))
print(calculate_bits(eights, uniform_random))

print("ensemble")

uniform_random_long = np.random.randint(0, 2, num_memories*2)

six_eights = np.vstack((sixes, eights))
zero_ones = np.vstack((zeros, ones))

print(calculate_bits(zero_ones, uniform_random_long))
print(calculate_bits(six_eights, uniform_random_long))