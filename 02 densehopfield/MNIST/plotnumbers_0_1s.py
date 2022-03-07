import torchvision
import numpy as np
import matplotlib.pyplot as plt

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

train_subset = train[0:12000, :-1]  # 6000 in each class
train_subset_labels = np.array(train[0:12000, -1], dtype=np.float64)

print(np.unique(train_subset_labels)) # returns [0, 1]

#convert data into binary data (white: 1 (all values bigger than 128), black: -1)
train_binary_subset = np.array(np.where(train_subset >= 128, 1, -1), dtype=np.float64)
print(train_binary_subset.shape, train_subset_labels.shape) #12000 x 784 dims, #12000 1-D array of labels

#IMAGES ALSO OK, RESHAPED (in 28 / 28) SHOWN AS CORRECT

#Select the first 50 images from the dataset (zeros and ones respective)
selected_0s = np.take(train_binary_subset, range(0,50), axis=0)
selected_1s = np.take(train_binary_subset, range(6000,6050), axis=0)

train_set = np.concatenate((selected_0s, selected_1s))

num_images = 100

num_row = 10
num_col = 10
# plot images
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(num_images):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(np.reshape(train_set[i], (28,28)), cmap='gray')
plt.tight_layout(pad=0.5)
plt.show()