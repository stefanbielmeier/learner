import numpy as np
from root.experiments.MNIST.digits.subsets import get_first_fifty_images
from root.experiments.MNIST.digits.subsets import make_binary


zeros, ones, sixes, eights = get_first_fifty_images(
    inBinary=True, zeroOnes=True)

num_neurons = 784
num_memories = 50

uniform_random = np.reshape(np.random.randint(
    0, 2, num_memories*num_neurons), (num_memories, num_neurons))

#Hamming distance is inverse similarity. Smaller distance means more similar. Harder to learn!
#So for hamming distance, we want to check how many bits need to be flipped to get from one
#pattern to any other one.
#The task of a Hopfield Network is to flip the bits in the input to get to the closest pattern.
#What is the pattern furthest away from any input pattern but can still be restored?
#How many bits have to be flipped to get from the noisy-most representation of a Hamming
#That is the capacity of the Hopfield Network.


def hamming_distance(string1, string2):
    # O(n) runtime
    dist_counter = 0
    for n in range(len(string1)):
        if string1[n] != string2[n]:
            dist_counter += 1
    return dist_counter


def hamming_distance(code):
    # O(N^2) runtime with N is number of patterns in the data
    dataset = code
    bit_divs = []
    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[0]):
            if i != j:
                bit_divs.append(hamming_distance(dataset[i, :], dataset[j, :]))

    bit_divs = np.array(bit_divs, dtype=np.float64)
    return np.min(bit_divs)

def mean_hamming_distance_in(code):
    dataset = code
    bit_divs = []
    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[0]):
            if i != j:
                bit_divs.append(hamming_distance(dataset[i, :], dataset[j, :]))

    bit_divs = np.array(bit_divs, dtype=np.float64)
    return np.mean(bit_divs)

def calculate_error_correction(dataset):
    return error_correction(hamming_distance(dataset))

def error_correction(hamming_distance):
    return (hamming_distance-1)/2

distance0 = hamming_distance(zeros)
distance1 = hamming_distance(ones)
distance6 = hamming_distance(sixes)
distance8 = hamming_distance(eights)

print(distance0)
print(distance1)
print(distance6)
print(distance8)

zero_ones = np.vstack((zeros, ones))
six_eights = np.vstack((sixes, eights))

distance_01 = hamming_distance(zero_ones)
distance_68 = hamming_distance(six_eights)
distance_uni = hamming_distance(uniform_random)

print(distance_01)
print(distance_68)
print(distance_uni)

xor = np.array([[-1,-1,-1],[-1,1,1],[1,-1,1],[1,1,-1]], dtype=np.float32)
xand = np.array([[-1,-1,1],[-1,1,-1],[1,-1,-1],[1,1,1]], dtype=np.float32)

xand_distance = hamming_distance(xand)
xor_distance = hamming_distance(xor)

print("XOR distance", xand_distance)
print("XAND distance", xand_distance)

#19 bits is the distance difference between the two bottlenecks of the datasets 
#the 2 most similar patterns from the dataset are the bottleneck (if we want 100% accuracy)
