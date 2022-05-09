from operator import pos
import matplotlib.pyplot as plt
import numpy as np
import torchvision

from scipy.stats import entropy

from root.infocapacity.estimate_cap import estimate_cap

num_neurons = 784
num_memories = 100

mnist_train = torchvision.datasets.MNIST(
    'dataset/', train=True, download=False)
mnist_test = torchvision.datasets.MNIST(
    'dataset/', train=False, download=False)

#convert data and labels into numpy arrays
train_set_array = mnist_train.data.numpy()
train_set_labels = mnist_train.targets.numpy()

test_set_array = mnist_test.data.numpy()
test_set_labels = mnist_test.targets.numpy()

#add labels to data
train_set = train_set_array.reshape(train_set_array.shape[0], -1) #flatten array
train_labels = train_set_labels

train = np.vstack((train_set.T, train_labels)).T

#TRANSFORMING is NOT THE PROBLEM (images stay the same)

#sort by labels
train = train[train[:, -1].argsort()]

train_subset_6 = train[36017:41935, :-1]
train_subset_8 = train[48200:54030, :-1]

train_subset_labels_6 = np.array(train[36017:41935, -1], dtype=np.float64) #36017 is the first 6 in the sorted set, and #41934 the last
train_subset_labels_8 = np.array(train[48200:54030, -1], dtype=np.float64) #48200 is the first 8 in the sorted set 

train_subset = np.concatenate((train_subset_6, train_subset_8))
train_subset_labels = np.concatenate((train_subset_labels_6, train_subset_labels_8))

print(np.unique(train_subset_labels)) # returns [6, 8]

#convert data into binary data (white: 1 (all values bigger than 128), black: -1)
train_binary_subset = np.array(np.where(train_subset >= 128, 1, -1), dtype=np.float64)

#Select the first 50 images from the dataset (sixes and eigth respective)

num_6s = train_subset_labels_6.shape[0]

selected_6s = np.take(train_binary_subset, range(0,50), axis=0)
selected_8s = np.take(train_binary_subset, range(num_6s+1,num_6s+1+50), axis=0)

selected_labels_6 = np.take(train_subset_labels_6, range(0,50), axis=0)
selected_labels_8 = np.take(train_subset_labels_8, range(0,50), axis = 0)

dataset_share = np.array([0.02, 0.04, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) #0.04 is 2 images in each "class"
x_axis = dataset_share * 100

information = []
six_eight_thresholds = []


for share in dataset_share:
    six_subset = selected_6s[0:int(share*50), :]
    eight_subset = selected_8s[0:int(share*50), :]

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




