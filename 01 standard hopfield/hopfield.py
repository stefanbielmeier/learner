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
    
    #async update, one neuron after the other, binary values only
    def update(self):
        for idx in range(self.activity.shape[0]):
            activation = np.dot(self.weights[idx,:], self.activity)
            if activation >= 0:
                self.activity[idx] = 1
            else:
                self.activity[idx] = -1
        return self.activity

    def learn(self, memories):
        for idx in range(self.weights.shape[0]):
            for jdex in range(self.weights.shape[1]):
                if idx != jdex:
                    result = 0
                    for mem in range(memories.shape[0]):
                        result += memories[mem,idx] * memories[mem,jdex]
                    self.weights[idx, jdex] = result
                
    def __str__(self):
        return str(self.activity)

def main():
    #capacity to store memories in Hopfield net is ~0.138 * neurons (n/2*log2())
    #For XOR / 4 memories with 3 features
    xor = np.array([[-1,-1,-1],[-1,1,1],[1,-1,1],[1,1,-1]])
    network = HopfieldNetwork(xor[0].shape[0])
    print(network)
    network.update()
    network.learn(xor)
    print(network)

if __name__ == "__main__":
    main()


