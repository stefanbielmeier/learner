
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_img(img, dim):
    plt.figure(figsize=(dim, dim))
    w_mat = plt.imshow(img, cmap=cm.coolwarm)
    plt.colorbar(w_mat)
    plt.title("Img")
    plt.tight_layout()
    plt.show()


def make_random_dataset(num_memories, num_neurons, zeroOnes = False):
    dataset = np.reshape(np.random.randint(0, 2, num_memories*num_neurons), (num_memories, num_neurons))
    if zeroOnes:
        return dataset
    else:
        return np.where(dataset == 0, -1, dataset)