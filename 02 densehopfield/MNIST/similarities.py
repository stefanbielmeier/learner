import numpy as np
import torchvision
from matplotlib import pyplot as plt
from scipy.stats import entropy

##GET MNIST

def kl_divergence(p, q):
    return np.sum(np.where(p == 0, p*np.log2(p/q), 0))

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

#sort by labels
train = train[train[:, -1].argsort()]

#create subset of 0s and 1s
train_subset = train[0:12000, :-1]  # 6000 in each class
#make binary
train_binary_subset = np.array(np.where(train_subset >= 128, 1, 0), dtype=np.int32)
#select 0s and 1s
selected_0s = np.take(train_binary_subset, range(0,50), axis=0)
selected_1s = np.take(train_binary_subset, range(6000,6050), axis=0)

train_0s_1s = np.concatenate((selected_0s, selected_0s))

#create subset of 6s and 8s
train_subset_6 = train[36017:41935, :-1]
train_subset_8 = train[48200:54030, :-1]

train_subset = np.concatenate((train_subset_6, train_subset_8))
train_binary_subset = np.array(np.where(train_subset >= 128, 1, 0), dtype=np.int32)

num_6s = train_subset_6.shape[0]

selected_6s = np.take(train_binary_subset, range(0,50), axis=0)
selected_8s = np.take(train_binary_subset, range(num_6s+1,num_6s+1+50), axis=0)

train_6s_8s = np.concatenate((selected_6s, selected_8s))

#get probability distributions
#probability of 1 in 6s
#probability of 1 in 8s

#problem: KL-divergence is not defined for probability of 1 = 0%

p_6s = np.sum(selected_6s, axis=0)
p_8s = np.sum(selected_8s, axis=0)
p_0s = np.sum(selected_0s, axis=0)
p_1s = np.sum(selected_1s, axis=0)

## get KL for 6s+8s vs. 0s+1s.
print("6s vs. 8s", kl_divergence(p_6s, p_8s))

#print KL for first 0 vs. second 0

##get KL for 6s vs. 8s

##get KL for 0s vs. 1s

