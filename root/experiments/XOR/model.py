import numpy as np
from root.experiments.MNIST.capacity.calcmemcap import get_memorization_capacity

from root.hopfieldnet.densehopfield import HopfieldNetwork

xor = np.array([[-1,-1,-1],[-1,1,1],[1,-1,1],[1,1,-1]])
xand = np.array([[0,0,1],[0,1,0],[1,0,0],[1,1,1]])

rows = xor.shape[0]
cols = xor.shape[1]

xorcap = get_memorization_capacity(xor)
print(xorcap)

xandcap = get_memorization_capacity(xand)
print(xandcap)