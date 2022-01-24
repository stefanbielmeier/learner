import numpy as np

num_classes = 2
num_neurons = 16

random = np.random.randint(0,2,16_000) 
randomarray = np.array(np.where(random == 0, -1, random), dtype=np.float64)

data = np.reshape(randomarray, (1_000,16))

duplicateddata = np.tile(data, 2)

print(data.shape)
print(duplicateddata.shape)
