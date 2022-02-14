import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

mnist_train = torchvision.datasets.MNIST('dataset/', train=True, download=False)
mnist_test = torchvision.datasets.MNIST('dataset/', train=False, download=False)

#convert data and labels into numpy arrays
train_set_array = mnist_train.data.numpy()
train_set_labels = mnist_train.targets.numpy()

test_set_array = mnist_test.data.numpy()
test_set_labels = mnist_test.targets.numpy()


#concat data with labels
train_set = train_set_array.reshape(train_set_array.shape[0], -1)
train_labels = train_set_labels

print(train_set.shape, train_labels.shape)
train = np.vstack((train_set.T, train_labels)).T

print(train.shape) #(60000 images, by 785 (784 pixels, and 1 label))

#sort by labels
train = train[train[:, -1].argsort()]

#create subset of two distinct classes, e.g. 0 and 1
train_subset = train[0:12000,:] #6000 in each class
print(np.unique(train_subset[:,-1])) #returns [0, 1]

#convert data into binary data (white: 1, black: -1)
train_binary = np.array(np.where(train_subset >= 128, 1, -1), dtype=np.float64)

#show example
plt.imshow(train_subset[0,:-1].reshape(28,28))
plt.gray()
plt.show()
plt.imshow(train_binary[0,:-1].reshape(28,28))
plt.gray()
plt.show()

#pick 1-200 random images from MNIST two-class dataset and use it to predict performance


