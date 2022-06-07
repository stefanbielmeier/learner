import numpy as np
from root.experiments.MNIST.digits.subsets import DATASET_PATH, get_first_fifty_images, get_subsets, get_training_data, make_binary
from root.experiments.MNIST.information.hamming import get_bottleneck_idxs, mean_hamming_distance, min_hamming_distance

training_data = get_training_data(DATASET_PATH)
training_data = make_binary(training_data, zeroOnes = False)
_, ones, _, _, _, _, _, _, _, _ = get_subsets(training_data)

first_500 = ones[:500, :]


bottleneck0s = get_bottleneck_idxs(first_500)[0]
hd0_mean = mean_hamming_distance(first_500)
hd0 = min_hamming_distance(first_500)

print(bottleneck0s)
print(hd0_mean)
print(hd0)

#TODO: Basically: create 80 datasets, and check what happens.

#TODO: verify the memorization accuracy and capacity depending on degree of n 