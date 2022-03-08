from random import gauss
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import entropy, norm

from minmemorypoly import get_memorization_capacity
from estimate_cap import estimate_cap

num_memories = 100
num_neurons = 16


rng = np.random.default_rng()
vals = rng.standard_normal(num_neurons)

gaussian_probabilities = np.empty(num_neurons)
for i in range(len(gaussian_probabilities)):
    gaussian_probabilities[i] = norm(0,1).pdf(vals[i])

print(gaussian_probabilities)
gaussian_probabilities.sort()
#plt.hist(gaussian_probabilities) #looks like data of 1s (furthest away from uniform random)
#plt.show()

binary_gaussian = np.zeros((100,16))

for col in range(len(gaussian_probabilities)):
    probability = gaussian_probabilities[col]
    number_1s = int(probability*binary_gaussian.shape[0])
    
    for one in range(number_1s):
        binary_gaussian[one, col] = 1

print(binary_gaussian)
uniform_random = np.reshape(np.random.randint(0, 2, num_memories*num_neurons), (num_memories, num_neurons))

p_binary_gaussian = np.sum(binary_gaussian, axis=0)
p_uniform_random = np.sum(uniform_random, axis=0)

x = np.arange(0,num_neurons)
fig, ax = plt.subplots()
scaled_p_binary_gaussian = np.divide(p_binary_gaussian, np.sum(p_binary_gaussian))
print(scaled_p_binary_gaussian)
print(gaussian_probabilities)

scaled_p_uniform_random = np.divide(p_uniform_random, np.sum(p_uniform_random))

print(scaled_p_binary_gaussian)

ax.bar(x+0.5, scaled_p_binary_gaussian, label='gaussian')
ax.bar(x, scaled_p_uniform_random, label='uniform random')

ax.legend(loc='best')
ax.set_title('Probability dist')

plt.show()

print("entropy", entropy(p_binary_gaussian, p_uniform_random))

#x percentage of the dataset analyzed
dataset_share = np.array([0.02, 0.04, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) #0.04 is 2 images in each "class"

min_memorization_polydegrees = []

#print("Num thresholds in 100 memories: ", estimate_cap(random))
"""



for share in dataset_share:
    
    selected = memories[0:int(share*num_memories), :]

    memorization_polydegree = get_memorization_capacity(selected)

    min_memorization_polydegrees.append(memorization_polydegree)


#plot
x_axis = dataset_share * 100
y2 = [2,4,10,20,30,40,50,60,70,80,90,100]

plt.plot(x_axis, min_memorization_polydegrees, label="capacity progression for % of dataset")
plt.plot(x_axis, y2, label="linear")

plt.xlabel("Percent of 100 images used as memories")
plt.ylabel("Capacity required to memorize (in terms of n)")

plt.legend(loc='best')
plt.show()


plt.plot(x_axis, min_memorization_polydegrees, label="capacity progression for % of dataset")
plt.plot(x_axis, y2, label="linear")

plt.xlabel("Percent of 100 images used as memories")
plt.ylabel("Capacity required to memorize (in terms of n)")

plt.legend(loc='best')

plt.ylim([0,20])
plt.show()
"""