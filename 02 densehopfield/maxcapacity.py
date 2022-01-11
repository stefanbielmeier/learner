"""
This code is based on Krotov and Hopfield 2016 paper in Neurips

This program calculates and plots the maximum number of memories a dense associatve Hopfield network can store
without retrieval error depending on two input factors:

N = Number of neurons in Hopfield net
n = degree of the polynomial smoothing function in the update function of the network
"""
import math

from scipy.special import factorial2
import matplotlib as plt

#calc capacity
def calc_k (num_neurons, poly_degree):
    factorial = factorial2(2*poly_degree-3)
    
    one = 1/(2*factorial)
    two = (num_neurons**(poly_degree-1))/(math.log(num_neurons))

    return one*two

print(calc_k(25,2))


