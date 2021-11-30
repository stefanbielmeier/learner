import sys
import os
import csv
import numpy as np


class HopfieldNetwork:
    def __init__(self, neurons):
        self.neurons = neurons
        
        #initialize activity as random
        a = np.random.randint(2, size=self.neurons) 
        a[a == 0] = -1
        self.activity = a
        
        #initialize symmetric matrix with random binary weights, and zeros
        matrix = np.random.randint(2, size=(self.neurons,self.neurons))
        matrix[matrix == 0] = -1
        matrix = np.tril(matrix) + np.tril(matrix,-1).T
        np.fill_diagonal(matrix, 0, wrap=False)
        self.weights = matrix

        self.__updateenergy__()

    def __updateenergy__(self):
        energy = 0
        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                if i != j:
                    energy = energy + self.weights[i, j] * self.activity[i] * self.activity[j]
        self.energy = energy * (-1/2)

    #async update, one neuron after the other, binary values only
    def __update__(self):
        for idx in range(self.activity.shape[0]):
            activation = np.dot(self.weights[idx,:], self.activity)
            if activation >= 0:
                self.activity[idx] = 1
            else:
                self.activity[idx] = -1
        self.__updateenergy__()
        return self.activity

    def learn(self, memories):
        for idx in range(self.weights.shape[0]):
            for jdex in range(self.weights.shape[1]):
                if idx != jdex:
                    result = 0
                    for mem in range(memories.shape[0]):
                        result = result + memories[mem,idx] * memories[mem,jdex]
                    self.weights[idx, jdex] = result / memories.shape[0]
        self.__update__()

    def predict(self, partialpattern):
        result = np.zeros(partialpattern.shape[0])
        for idx in range(partialpattern.shape[0]):
            activation = np.dot(self.weights[idx,:], partialpattern)
            if activation >= 0:
                result[idx] = 1
            else:
                result[idx] = -1
        return result 

    def __str__(self):
        return "State: " + str(self.activity) + "\n energy in network: " + str(self.energy)

def main():
    #capacity to store memories in Hopfield net is ~0.138 * neurons (n/2*log2())
    #For XOR / 4 memories with 3 features
    xor = np.array([[1,1,-1]])

    x = np.array([[[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1],[1,1,1,1,1],[-1,-1,-1,-1,-1]], 
    [[1,1,1,1,1],[-1,-1,-1,-1,-1],[1,1,1,1,1],[-1,-1,-1,-1,-1],[1,1,1,1,1]]])
    
    #flattens all dimensions except first dimension
    x = x.reshape(x.shape[0], -1)

    partialpattern = np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[-1,-1,-1,-1,-1],[1,1,1,1,1]]).flatten()

    network = HopfieldNetwork(25)
    print(network)
    network.learn(x)
    #sign blind
    print(network)

    print(network.predict(partialpattern))

if __name__ == "__main__":
    main()


