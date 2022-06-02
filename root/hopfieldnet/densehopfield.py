from cmath import tanh
import math

import numpy as np

from root.utils import plot_img

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

        self.continuous = continous

    def __smooth_function(self, x):
        #polynomial energy function (not rectified polynomial!) Hopfield & Krotov 2016
        if self.max_cap:
            return math.exp(x)
        else:
            return x**self.polydegree

    #async update, one neuron after the other, selected randomly, binary values only
    def update(self, state):
        self.excitation = state.copy()

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

        if self.continuous == False:
            if result >= 0:
                self.excitation[i] = 1 #i
            else:
                self.excitation[i] = -1
        
        if self.continuous == True:
            beta = 1/(self.polydegree)
            #use tanh as activation function :-)
            self.excitation[i] = np.tanh(beta*result)
        
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
    print("Use this Hopfield network in other files!")

if __name__ == "__main__":
    main()
