import torchvision
import numpy as np
import matplotlib.pyplot as plt

from estimate_cap import estimate_cap
from actualcap import get_recall_qualities


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

print(train.shape)  # (60000 images, by 785 (784 pixels, and 1 label))

#sort by labels
train = train[train[:, -1].argsort()]

#create subset of two distinct classes, e.g. 0 and 1
train_subset = train[0:12000, :-1]  # 6000 in each class
train_subset_labels = np.array(train[0:12000, -1], dtype=np.float64)

print(np.unique(train_subset_labels)) # returns [0, 1]

#convert data into binary data (white: 1 (all values bigger than 128), black: -1)
train_binary_subset = np.array(np.where(train_subset >= 128, 1, -1), dtype=np.float64)
print(train_binary_subset.shape, train_subset_labels.shape) #12000 x 784 dims, #12000 1-D array of labels

#IMAGES ALSO OK, RESHAPED (in 28 / 28) SHOWN AS CORRECT

class_cutoff = int(train_binary_subset.shape[0]/2)
dimensionality = train_binary_subset.shape[1]

#Select 100 images from the dataset, 50 zeros and 50 ones
random_0_indeces = np.random.randint(0, class_cutoff, 50)
random_1_indeces = np.random.randint(class_cutoff, class_cutoff*2, 50)

selected_0s = np.take(train_binary_subset, random_0_indeces, axis=0)
selected_1s = np.take(train_binary_subset, random_1_indeces, axis=0)

dataset = np.concatenate((selected_0s, selected_1s))
#also works

#x and y
polydegrees = np.array(range(1,2,1))
print(polydegrees)
accuracies = get_recall_qualities(dataset, polydegrees=polydegrees, num_neurons=dimensionality, plot_updated_images=False)

#plot
plt.plot(polydegrees, accuracies, label="accuracy progression for capacity")

plt.legend(loc='best')
plt.show()
