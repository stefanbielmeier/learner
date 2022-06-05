import numpy as np
from root.experiments.MNIST.digits.subsets import DATASET_PATH, get_first_fifty_images, get_subsets, get_training_data
from root.experiments.MNIST.information.hamming import get_bottleneck_idxs, min_hamming_distance


zeros, ones, twos, threes, fours, fives, sixes, sevens, eights, nines = get_first_fifty_images(inBinary=True)

bottleneck0s = get_bottleneck_idxs(zeros)[0]
bottleneck1s = get_bottleneck_idxs(ones)[0]
bottleneck2s = get_bottleneck_idxs(twos)[0]
bottleneck3s = get_bottleneck_idxs(threes)[0]
bottleneck4s = get_bottleneck_idxs(fours)[0]
bottleneck5s = get_bottleneck_idxs(fives)[0]
bottleneck6s = get_bottleneck_idxs(sixes)[0]
bottleneck7s = get_bottleneck_idxs(sevens)[0]
bottleneck8s = get_bottleneck_idxs(eights)[0]
bottleneck9s = get_bottleneck_idxs(nines)[0]

all_data = np.concatenate((zeros, ones, twos, threes, fours, fives, sixes, sevens, eights, nines))

print(bottleneck0s)
print(bottleneck1s)
print(bottleneck2s)
print(bottleneck3s)
print(bottleneck4s)
print(bottleneck5s)
print(bottleneck6s)
print(bottleneck7s)
print(bottleneck8s)
print(bottleneck9s)

hd_all = min_hamming_distance(all_data)
print(hd_all)