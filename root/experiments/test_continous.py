import numpy as np
import matplotlib.pyplot as plt

from root.hopfieldnet.densehopfield import HopfieldNetwork
from root.utils import plot_img

from actualcap import get_recall_qualities

num_classes = 2
num_neurons = 25

random = np.random.normal(0,0.25,16_000)
data = np.reshape(random,(int(16_000/num_neurons),num_neurons))

network = HopfieldNetwork(num_neurons, 5, continous=True)
network.learn(data)

plot_img(np.reshape(data[1], (5,5)),5)
network.update(data[1])

network.plot()
