
from root.experiments.MNIST.capacity.calcmemcap import get_memorization_capacity
from root.experiments.MNIST.digits.subsets import get_first_fifty_images
from root.experiments.MNIST.information.hamming import get_bottleneck_idxs, min_hamming_distance


zeros, ones, twos, threes, fours, fives, sixes, sevens, eights, nines = get_first_fifty_images(inBinary=True)

bottleneck1s = get_bottleneck_idxs(ones)[0]
print(min_hamming_distance(ones))
mem_ones = get_memorization_capacity(ones, recall_quality = 1.0, startAt = 38, test_idxs = bottleneck1s, verbose = True, corrupt=False, add_noise_bits=5)

print(mem_ones)
"""
restore performance 1.0
restore performance 0.9974489795918368
average restore performance 0.9987244897959184
current capacity:  52
restore performance 1.0
restore performance 0.9974489795918368
average restore performance 0.9987244897959184
current capacity:  53
restore performance 1.0
restore performance 1.0
average restore performance 1.0
53
"""