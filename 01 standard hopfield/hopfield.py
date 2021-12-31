import sys
import os
import csv
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class HopfieldNetwork:
    def __init__(self, neurons):
        self.neurons = neurons
        
        #initialize excitation of neurons as 0
        self.excitation = np.zeros(self.neurons) 
        
        #initialize symmetric matrix with 0 weights
        self.weights = np.zeros((self.neurons,self.neurons))

        self.__updateenergy__()

    def __updateenergy__(self):
        energy = 0
        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                if i != j:
                    energy = energy + self.weights[i, j] * self.excitation[i] * self.excitation[j]
        self.energy = energy * (-1/2)

    #async update, one neuron after the other, binary values only
    def update(self, state):
        for idx in range(state.shape[0]):
            activation = np.dot(self.weights[idx,:], state)
            if activation >= 0:
                self.excitation[idx] = 1
            else:
                self.excitation[idx] = -1
        self.__updateenergy__()
    
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

        plt.figure(figsize=(dims, dims))
        w_mat = plt.imshow(state, cmap=cm.coolwarm)
        plt.colorbar(w_mat)
        plt.title("Excitation of Network")
        plt.tight_layout()
        plt.show()


def plot_img(img, dim):
    plt.figure(figsize=(dim, dim))
    w_mat = plt.imshow(img, cmap=cm.coolwarm)
    plt.colorbar(w_mat)
    plt.title("Img")
    plt.tight_layout()
    plt.show()

def main():
    #capacity to store memories in Hopfield net is ~0.138 * neurons (n/2*log2())
    #For XOR / 4 memories with 3 features
    xor = np.array([[1,1,-1]])

    x = np.array([[[1,1,1,1,1],[-1,-1,1,-1,-1],[-1,-1,1,-1,-1],[-1,-1,1,-1,-1],[-1,-1,1,-1,-1]], 
    [[1,-1,-1,-1,1],[1,-1,-1,-1,1],[1,1,1,1,1],[1,-1,-1,-1,1],[1,-1,-1,-1,1]]])
    
    #flattens all dimensions except first dimension
    x = x.reshape(x.shape[0], -1)

    #three noise bits
    noisy_t = np.array([[1,-1,1,1,1],[-1,-1,1,-1,1],[-1,1,1,-1,-1],[-1,-1,1,-1,1],[-1,-1,1,-1,-1]])

    noisy_h = np.array([[-1,-1,-1,-1,1],[1,1,-1,-1,1],[1,1,1,1,1],[1,-1,-1,-1,1],[1,1,1,-1,1]])

    network = HopfieldNetwork(25)
    #network.plot()
    network.learn(x)
    network.update(x[0].flatten())
    print(network)


    plot_img(noisy_t, 5)
    network.update(noisy_t.flatten())
    network.plot()
    print(network)

    #sort of doesn't work yet – probably because it updates excitation
    plot_img(noisy_h, 5)
    network.update(noisy_h.flatten())
    network.plot()
    print(network)

if __name__ == "__main__":
    main()


