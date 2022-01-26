from cmath import tanh
import math
from re import I

import numpy as np

from utils import plot_img

class HopfieldNetwork:
    def __init__(self, neurons, polydegree, max_cap = False, continous = False):
        self.neurons = neurons
        
        #initialize excitation of neurons as 0
        self.excitation = np.zeros(self.neurons) 
        
        #store memories
        self.memories = 0

        self.energy = 0
        
        self.polydegree = polydegree

        self.max_cap = max_cap

        self.continuous = True

    def __smooth_function(self, x):
        #polynomial energy function (not rectified polynomial!) Hopfield & Krotov 2016
        if self.max_cap:
            return math.exp(x)
        else:
            return x**self.polydegree

    #async update, one neuron after the other, selected randomly, binary values only
    def update(self, state):
        self.excitation = state

        for i in range(self.neurons):
            result = 0
            for memory in self.memories:    
                
                jsum = 0
                for j in range(self.neurons):
                    if i != j:
                        jsum = jsum + self.excitation[j] * memory[j]

                result = result + (self.__smooth_function(1 * memory[i] + jsum) - self.__smooth_function(-1 * memory[i] + jsum))

            self.__activation_function(result, i)

    def __activation_function(self, result, i):
        beta = 1/(self.polydegree)

        if self.continuous == False:
            if result >= 0:
                self.excitation[i] = 1 #i
            else:
                self.excitation[i] = -1
        
        if self.continuous == True:
            #use tanh as activation function :-)
            self.excitation[i] = np.tanh(beta*result)
            print(result)
        
    def __str__(self):
        return "Energy of Network:" + str(self.energy)

    def learn(self, memories):
        self.memories = memories

    def plot(self):
        dims = int(math.sqrt(self.neurons))
        state = self.excitation.reshape(dims,dims)
        plot_img(state, dims)

    def get_state(self):
        return self.excitation

def main():    
    T = np.array([[1,1,1,1,1],[-1,-1,1,-1,-1],[-1,-1,1,-1,-1],[-1,-1,1,-1,-1],[-1,-1,1,-1,-1]])
    H = np.array([[1,-1,-1,-1,1],[1,-1,-1,-1,1],[1,1,1,1,1],[1,-1,-1,-1,1],[1,-1,-1,-1,1]])
    E = np.array([[1,1,1,1,1], [1,-1,-1,-1,-1,], [1,1,1,1,1], [1,-1,-1,-1,-1], [1,1,1,1,1]])

    S = np.array([[1,1,1,1,1], [1,-1,-1,-1,-1,], [1,1,1,1,1], [-1,-1,-1,-1,1], [1,1,1,1,1]])

    #ten noisy bits
    noisy_t = np.array([[-1,-1,1,1,1],[-1,-1,1,-1,1],[-1,1,1,-1,-1],[-1,1,1,-1,1],[1,-1,1,1,1]])

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
    
    newnet = HopfieldNetwork(25, 3)
    newnet.learn(fourmems)

    noisy_s = np.array([[1,1,1,-1,1], [-1,1,-1,-1,-1], [1,1,1,1,1], [-1,-1,1,-1,1], [1,1,1,1,1]])

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

    """
    print('S')
    plot_img(noisy_s, 5)
    newnet.update(noisy_s.flatten())
    newnet.plot()
    """
    

if __name__ == "__main__":
    main()
