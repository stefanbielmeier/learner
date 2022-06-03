import contextlib
import numpy as np
import matplotlib.pyplot as plt

from root.experiments.MNIST.capacity.definitions import DATASET_SHARE
from root.experiments.MNIST.capacity.calcmemcap import calc_memorization_capacities, get_memorization_capacity
from root.experiments.MNIST.digits.subsets import get_first_fifty_images
from root.experiments.MNIST.information.hamming import get_bottleneck_idxs

zeros, ones, sixes, eights = get_first_fifty_images(inBinary = True)

zero_ones = np.vstack((zeros, ones))
six_eights = np.vstack((sixes, eights))

bottleneck01 = get_bottleneck_idxs(zero_ones)
bottleneck68 = get_bottleneck_idxs(six_eights)
 
file_path = 'test_others_now2.txt'
with open(file_path, "w") as o:
    with contextlib.redirect_stdout(o):
        mem_zero_ones = get_memorization_capacity(zero_ones, recall_quality = 1.0, startAt = 40, test_idxs = []) #test random data.
        mem_six_eights = get_memorization_capacity(six_eights, recall_quality = 1.0, startAt = 17, test_idxs = []) #test random data.