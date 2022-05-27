import numpy as np
from root.experiments.MNIST.digits.subsets import get_first_fifty_images
from root.experiments.MNIST.digits.subsets import make_binary



zeros, ones, sixes, eights = get_first_fifty_images(inBinary=True, zeroOnes=True)

num_neurons = 784
num_memories = 50

uniform_random = np.reshape(np.random.randint(
    0, 2, num_memories*num_neurons), (num_memories, num_neurons))

def hamming_distance(string1, string2):
	dist_counter = 0
	for n in range(len(string1)):
		if string1[n] != string2[n]:
			dist_counter += 1
	return dist_counter


def calculate_bits(dataset, uniform_random):
    bit_divs = []
    for row_num in range(dataset.shape[0]):
        bit_divs.append(hamming_distance(dataset[row_num,:], uniform_random[row_num,:]))

    bit_divs = np.array(bit_divs, dtype=np.float64)

    return np.sum(bit_divs)

print(calculate_bits(zeros, uniform_random))
print(calculate_bits(ones, uniform_random))
print(calculate_bits(sixes, uniform_random))
print(calculate_bits(eights, uniform_random))
