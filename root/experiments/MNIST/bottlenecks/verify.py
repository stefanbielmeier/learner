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

hd1 = min_hamming_distance(zeros)
hd2 = min_hamming_distance(ones)
hd3 = min_hamming_distance(twos)
hd4 = min_hamming_distance(threes)
hd5 = min_hamming_distance(fours)
hd6 = min_hamming_distance(fives)
hd7 = min_hamming_distance(sixes)
hd8 = min_hamming_distance(sevens)
hd9 = min_hamming_distance(eights)
hd10 = min_hamming_distance(nines)

print(hd1)
print(hd2)
print(hd3)
print(hd4)
print(hd5)
print(hd6)
print(hd7)
print(hd8)
print(hd9)
print(hd10)

hd_all = min_hamming_distance(all_data)