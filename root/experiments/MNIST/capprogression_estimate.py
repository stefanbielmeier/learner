import matplotlib.pyplot as plt
import numpy as np
import torchvision

from scipy.stats import entropy

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

#create subset of two distinct classes, e.g. 0 and 1
train_subset = train[0:12000, :-1]  # 6000 in each class
train_subset_labels = np.array(train[0:12000, -1], dtype=np.float64)

#convert data into binary data (white: 1 (all values bigger than 128), black: -1)
train_binary_subset = np.array(np.where(train_subset >= 128, 1, -1), dtype=np.float64)

#Select the first 50 images from the dataset (zeros and ones respective)
selected_0s = np.take(train_binary_subset, range(0,50), axis=0)
selected_1s = np.take(train_binary_subset, range(6000,6050), axis=0)

dataset_share = np.array([0.02, 0.04, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) #0.04 is 2 images in each "class"
x_axis = dataset_share * 100

information = []

for share in dataset_share:
    zero_subset = selected_0s[0:int(share*50), :]
    one_subset = selected_1s[0:int(share*50), :]

    partial_zero_ones_unary = np.concatenate((zero_subset, one_subset))
    partial_zero_ones_binary = np.array(np.where(partial_zero_ones_unary == -1, 0, partial_zero_ones_unary), dtype=np.float64)

    p_zero_ones = np.sum(partial_zero_ones_binary, axis=0)

    info_content = entropy(p_zero_ones, base=2)
    information.append(info_content)

mem_cap = np.array([2, 4, 8, 12, 16, 20, 21, 25, 24, 26, 29, 27])
info_gain = np.array([1.8828204158228377, 1.6758262676616411, 1.4892695268950995, 1.4081247313418512, 1.4022956608563981, 1.4205585227111361, 1.4130696261585378, 1.3961868620537718, 1.380697627864883, 1.3682492575455916, 1.3730425461464963, 1.3741251006585482])
flipped_info_gain = info_gain[0]*6 + -4.5 * np.array([1.8828204158228377, 1.6758262676616411, 1.4892695268950995, 1.4081247313418512, 1.4022956608563981, 1.4205585227111361, 1.4130696261585378, 1.3961868620537718, 1.380697627864883, 1.3682492575455916, 1.3730425461464963, 1.3741251006585482]) 
random_thresholds = np.array([1.0, 2.0, 2.807354922057604, 3.584962500721156, 4.169925001442312, 4.754887502163468, 5.0, 5.129283016944966, 5.247927513443585, 5.357552004618084, 5.491853096329675, 5.614709844115208])
zero_one_thresholds = np.array([1.0, 2.0, 3.321928094887362, 4.247927513443585, 4.754887502163468, 5.129283016944966, 5.459431618637297, 5.700439718141092, 5.857980995127572, 6.0, 6.129283016944966, 6.266786540694901])


#Print information
print(zero_one_thresholds*6)
print(mem_cap)

##plotting the stuff again

fig, ax = plt.subplots()

x_axis = dataset_share * 100

ax.plot(x_axis, mem_cap, label='memorization capacity', color='cyan')
ax.set_xlabel("number of memories")
ax.set_ylabel("memorization capacity")
ax.legend()

ax2 = ax.twinx()
#ax2.plot(x_axis, flipped_info_gain, label='KL divergence')
ax2.plot(x_axis, random_thresholds, label='# of Thresholds in Uniform Random Data')
ax2.plot(x_axis, zero_one_thresholds, label='# of Thresholds in Zero or One Data')

ax2.set_ylabel("Thresholds and KL Divergence in bits")
ax2.legend()

plt.show()




