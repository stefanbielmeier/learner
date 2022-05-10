from operator import pos
import matplotlib.pyplot as plt
import numpy as np
import torchvision

from scipy.stats import entropy

from root.infocapacity.estimate_cap import estimate_cap

from root.experiments.MNIST.digits.subsets import get_first_fifty_images

num_neurons = 784
num_memories = 100

_,_, sixes, eights = get_first_fifty_images(inBinary = True)

dataset_share = np.array([0.02, 0.04, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) #0.04 is 2 images in each "class"
x_axis = dataset_share * 100

information = []
six_eight_thresholds = []

for share in dataset_share:
    six_subset = sixes[0:int(share*50), :]
    eight_subset = eights[0:int(share*50), :]

    partiail_6_8s_unary = np.concatenate((six_subset, eight_subset))
    partial_6_8s_binary = np.array(np.where(partiail_6_8s_unary == -1, 0, partiail_6_8s_unary), dtype=np.float64)

    p_six_eights = np.sum(partial_6_8s_binary, axis=0)

    threshold, _ = estimate_cap(partiail_6_8s_unary)
    six_eight_thresholds.append(threshold)

mem_cap_6and8 = np.array([2,4,7,9,11,12,13,13,14,14,13,15])
mem_cap_0and1 = np.array([2, 4, 8, 12, 16, 20, 21, 25, 24, 26, 29, 27])

arbitrary_scaling_factor = 1

random_thresholds = arbitrary_scaling_factor * np.array([1.0, 2.0, 2.807354922057604, 3.584962500721156, 4.169925001442312, 4.754887502163468, 5.0, 5.129283016944966, 5.247927513443585, 5.357552004618084, 5.491853096329675, 5.614709844115208])
zero_one_thresholds = arbitrary_scaling_factor * np.array([1.0, 2.0, 3.321928094887362, 4.247927513443585, 4.754887502163468, 5.129283016944966, 5.459431618637297, 5.700439718141092, 5.857980995127572, 6.0, 6.129283016944966, 6.266786540694901])
print(six_eight_thresholds)

six_eight_thresholds = np.array(six_eight_thresholds)

fig, ax = plt.subplots()

x_axis = dataset_share * 100

ax.plot(x_axis, mem_cap_6and8, label='mem cap 6 and 8', color='cyan')
ax.plot(x_axis, mem_cap_0and1, label='mem cap 0s and 1s', color='blue')
ax.set_xlabel("number of memories")
ax.set_ylabel("memorization capacity")
ax.set_ylim(0, 30)
ax.legend(loc='best')

ax2 = ax.twinx()
#ax2.plot(x_axis, flipped_info_gain, label='KL divergence')
ax2.plot(x_axis, random_thresholds, label='# of Thresholds in Uniform Random Data')
ax2.plot(x_axis, zero_one_thresholds, label='# of Thresholds in Zero or One Data')
ax2.plot(x_axis, six_eight_thresholds, label='# of Thresholds in Six or Eight Data')

ax2.set_ylabel("Thresholds and KL Divergence in bits")
ax2.set_ylim(0, 30)
ax2.legend(loc='lower right')

fig.suptitle('Memorization Capacity and Number of Thresholds')

plt.show()




