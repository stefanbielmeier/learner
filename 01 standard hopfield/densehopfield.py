import math

import numpy as np

from utils import plot_img

class HopfieldNetwork:
    def __init__(self, neurons):
        self.neurons = neurons
        
        #initialize excitation of neurons as 0
        self.excitation = np.zeros(self.neurons) 
        
        #initialize symmetric matrix with 0 weights
        self.weights = np.zeros((self.neurons,self.neurons))

        self.energy = 0

        self.__updateenergy()

        #capacity to store memories in binary Hopfield net is ~0.138 * neurons (n/2*log2()), assuming completely random memories
        print("Capacity of network (#memories): {}".format(self.neurons*0.138))


    #using 
    def __updateenergy(self):
        energy = 0
        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                if i != j:
                    energy = energy + self.weights[i, j] * self.excitation[i] * self.excitation[j]
        self.energy = energy * (-1/2)

    #async update, one neuron after the other, selected randomly, binary values only
    def update(self, state):
        self.excitation = state
        self.__updateenergy()
        
        idxs = np.random.permutation(state.shape[0])
        
        for idx in idxs:
            print(self)
            activation = np.dot(self.weights[idx,:], state)
            if activation >= 0:
                self.excitation[idx] = 1
            else:
                self.excitation[idx] = -1
            self.__updateenergy()
    
    def __str__(self):
        return "Energy of Network:" + str(self.energy)

    def learn(self, memories):
        for idx in range(self.weights.shape[0]):
            for jdex in range(self.weights.shape[1]):
                if idx != jdex:
                    result = 0
                    for mem in range(memories.shape[0]):
                        result = result + memories[mem,idx] * memories[mem,jdex]
                    self.weights[idx, jdex] = result / memories.shape[0]

    def plot(self):
        dims = int(math.sqrt(self.neurons))
        state = self.excitation.reshape(dims,dims)
        plot_img(state, dims)

def main():    
    T = np.array([[1,1,1,1,1],[-1,-1,1,-1,-1],[-1,-1,1,-1,-1],[-1,-1,1,-1,-1],[-1,-1,1,-1,-1]])
    H = np.array([[1,-1,-1,-1,1],[1,-1,-1,-1,1],[1,1,1,1,1],[1,-1,-1,-1,1],[1,-1,-1,-1,1]])
    E = np.array([[1,1,1,1,1], [1,-1,-1,-1,-1,], [1,1,1,1,1], [1,-1,-1,-1,-1], [1,1,1,1,1]])

    #ten noisy bits
    noisy_t = np.array([[-1,-1,1,1,1],[-1,-1,-1,-1,1],[-1,1,1,-1,-1],[-1,1,1,-1,1],[1,-1,1,1,1]])

    #three noise bits can be restored with two or three memories – no problem
    noisy_h = np.array([[-1,-1,-1,-1,1],[1,1,-1,-1,1],[1,1,1,1,1],[1,-1,-1,-1,1],[1,1,1,-1,1]])

    #six noise bits and little overlap => Corrupted H
    noisy_e = np.array([[1,1,-1,1,1], [-1,1,-1,1,-1], [1,1,1,1,1], [-1,-1,-1,-1,-1], [1,1,-1,1,1]])

    #adding a fourth memory makes the network fail and get really random / noisy results!
    X = np.array([[1,-1,-1,-1,1], [-1,1,-1,1,-1], [-1,-1,1,-1,-1], [-1,1,-1,1,-1], [1,-1,-1,-1,1]])
    #six noise bits
    noisy_x = np.array([[-1,-1,-1,-1,1], [-1,-1,-1,1,-1], [1,1,1,-1,1], [1,1,-1,1,-1], [1,-1,-1,1,-1]])

    fourmems = np.stack([T,H,E,X], axis=0)
    fourmems = fourmems.reshape(fourmems.shape[0],-1) #flattens all except first dim, => 2D matrix
    
    newnet = HopfieldNetwork(25)
    newnet.learn(fourmems)

    print('T')
    plot_img(noisy_t, 5)
    newnet.update(noisy_t.flatten())
    newnet.plot()

    print('H')
    plot_img(noisy_h, 5)
    newnet.update(noisy_h.flatten())
    newnet.plot()

    print('E')
    plot_img(noisy_e, 5)
    newnet.update(noisy_e.flatten())
    newnet.plot()

    print('X')
    plot_img(noisy_x, 5)
    newnet.update(noisy_x.flatten())
    newnet.plot()

if __name__ == "__main__":
    main()


