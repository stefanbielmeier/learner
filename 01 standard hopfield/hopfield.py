import sys
import os
import csv
import numpy as np



class Network:
    def __init__(self, neurons):
        self.neurons = neurons
        self.weights = np.zeros((self.neurons, self.neurons))
    
    def train(self, memories):
        pass

    def predict(self):
        pass

def main():
    print("Hi")
    
    network = Network(neurons)
    network.train(memories)


if __name__ == "__main__":
    main()


