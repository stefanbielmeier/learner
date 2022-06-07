

from root.experiments.MNIST.capacity.calcmemcap import get_memorization_capacity
from root.utils import make_random_dataset


dataset = make_random_dataset(100, 784, zeroOnes = False)

#can't add more than (0.43*num_neurons)-1 bits, otherwise: higher capacity needed or it doesn't work at all!
memcap = get_memorization_capacity(dataset, recall_quality=1.0, verbose=True, startAt=3, test_idxs=[1,2,3], corrupt=False, add_noise_bits=340) 

print(memcap)