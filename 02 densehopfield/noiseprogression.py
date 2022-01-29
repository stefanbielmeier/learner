#todo :-D

import numpy as np
from actualcap import get_recall_qualities
import matplotlib.pyplot as plt


num_neurons = 16
num_memories = 100

random = np.random.randint(0,2,num_memories) 
randomarray = np.array(np.where(random == 0, -1, random), dtype=np.float64)

#in-class noise percentage 
for noise_percentage in [0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]:
    mems_900 = np.tile(randomarray, 9)
    
    # creates noise_percentage bits of noise in the copied data
    noise = np.array([0.] * num_neurons*900*(1-noise_percentage) + [1.] * num_neurons*900*(1-0.5*noise_percentage) + [-1.] *num_neurons*900*(1-0.5*noise_percentage)) 
    np.random.shuffle(noise) #shuffle noisy bits
    noisydata = np.where(noise == 0, mems_900, noise)

    mems_1000 = np.concat(randomarray, mems_900)
    data = np.reshape(randomarray, (1_000,16)) 


    polydegrees = np.arange(1,30) #m
    recall_qualities = get_recall_qualities(data, polydegrees, num_neurons)
    plt.plot(polydegrees, recall_qualities, label="noise percentage in copies: {}".format(noise_percentage))

plt.legend(loc="best")
plt.show()
