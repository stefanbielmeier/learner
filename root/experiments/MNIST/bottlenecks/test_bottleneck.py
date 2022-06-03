import contextlib
import numpy as np
import matplotlib.pyplot as plt

from root.experiments.MNIST.capacity.calcmemcap import get_memorization_capacity
from root.experiments.MNIST.digits.subsets import get_first_fifty_images
from root.experiments.MNIST.information.hamming import get_bottleneck_idxs

zeros, ones, sixes, eights = get_first_fifty_images(inBinary = True)

zero_ones = np.vstack((zeros, ones))
six_eights = np.vstack((sixes, eights))

bottleneck01 = get_bottleneck_idxs(zero_ones)[0]
bottleneck68 = get_bottleneck_idxs(six_eights)[0]

dataset1 = zero_ones[bottleneck01, :]
dataset2 = six_eights[bottleneck68, :]
 
file_path = 'test_bottleneck_only.txt'
with open(file_path, "w") as o:
    with contextlib.redirect_stdout(o):
        mem_zero_ones = get_memorization_capacity(dataset1, recall_quality = 1.0, startAt = 2, test_idxs = np.array([0,1])) #test the two bottleneck files
        mem_six_eights = get_memorization_capacity(dataset2, recall_quality = 1.0, startAt = 2, test_idxs = np.array([0,1])) #test the two bottleneck files