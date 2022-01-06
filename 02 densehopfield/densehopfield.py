import math

import numpy as np

from utils import plot_img

class HopfieldNetwork:
    def __init__(self, neurons):
        self.neurons = neurons
        
        #initialize excitation of neurons as 0
        self.excitation = np.zeros(self.neurons) 
        
        #store memories
        self.memories = 0

        self.energy = 0

    def __smooth_function(self, x):
        
        return math.exp(x)

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

            if result >= 0:
                self.excitation[i] = 1 #i
            else:
                self.excitation[i] = -1
        
    def __str__(self):
        return "Energy of Network:" + str(self.energy)

    def learn(self, memories):
        self.memories = memories

    def plot(self):
        dims = int(math.sqrt(self.neurons))
        state = self.excitation.reshape(dims,dims)
        plot_img(state, dims)

def main():    
    T = np.array([[1,1,1,1,1],[-1,-1,1,-1,-1],[-1,-1,1,-1,-1],[-1,-1,1,-1,-1],[-1,-1,1,-1,-1]])
    H = np.array([[1,-1,-1,-1,1],[1,-1,-1,-1,1],[1,1,1,1,1],[1,-1,-1,-1,1],[1,-1,-1,-1,1]])
    E = np.array([[1,1,1,1,1], [1,-1,-1,-1,-1,], [1,1,1,1,1], [1,-1,-1,-1,-1], [1,1,1,1,1]])

    S = np.array([[1,1,1,1,1], [1,-1,-1,-1,-1,], [1,1,1,1,1], [-1,-1,-1,-1,1], [1,1,1,1,1]])

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

    fourmems = np.stack([T,H,E,X, S], axis=0)
    fourmems = fourmems.reshape(fourmems.shape[0],-1) #flattens all except first dim, => 2D matrix
    
    newnet = HopfieldNetwork(25)
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

    print('S')
    plot_img(noisy_s, 5)
    newnet.update(noisy_s.flatten())
    newnet.plot()

if __name__ == "__main__":
    main()


