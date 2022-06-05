from enum import Enum
import os

import matplotlib.pyplot as plt
import numpy as np
import torchvision

from root.definitions import ROOT_DIR

zero_start_index = 0
zero_end_index = 5999
one_start_index = 6000
one_end_index = 11999

six_start_index = 36017
six_end_index = 41935
eight_start_index = 48200
eight_end_index = 54030

DATASET_PATH = os.path.join(ROOT_DIR, "experiments", "MNIST", "dataset")

#subroutine
def get_training_data(path):
    """
    @returns
    sorted training data as a numpy array
    in binary [-1,1],
    labels in last column
    """

    mnist_train = torchvision.datasets.MNIST(
        path, train=True, download=False)

    #convert data and labels into numpy arrays
    train_set_array = mnist_train.data.numpy()
    train_set_labels = mnist_train.targets.numpy()

    #add labels to data
    train_set = train_set_array.reshape(
        train_set_array.shape[0], -1)  # flatten array
    train_labels = train_set_labels

    #stack
    train = np.vstack((train_set.T, train_labels)).T

    #sort by labels
    train = train[train[:, -1].argsort()]

    return np.array(train, dtype=np.float64)

#subroutine
def make_binary(data, zeroOnes = False):
    """
    @param: numpy 2d array of training data, last column is labels
    @return: numpy 2d array of training data, last column is labels
    """
    labels = data[:, -1]
    set = data[:,:-1]
    
    binary = np.array(np.where(set >= 128, 1, -1), dtype=np.float64)
    if zeroOnes:
        binary = np.where(binary == -1, 0, 1)

    return np.vstack((binary.T, labels)).T

#subroutine

def find_class_start_end_indeces(training_data):
    """
    In dataset: find the last index of a class and return it
    """

    start_indeces = {0: 0}
    curr = 0
    for row_idx in range(training_data.shape[0]):
        if training_data[row_idx,-1] != curr:
            start_indeces[training_data[row_idx,-1]] = row_idx
            curr = training_data[row_idx,-1]
    
    return start_indeces

def get_subsets(all_digits):
    """
    @param: numpy 2D array of training data, last column is labels
    @return: object of Digit subsets of MNIST, as numpy arrays with 50 elements each 
    """
    start_indexes = find_class_start_end_indeces(all_digits)

    zeros_data = all_digits[start_indexes[0]:start_indexes[1], :]
    ones_data = all_digits[start_indexes[1]:start_indexes[2], :]
    twos_data = all_digits[start_indexes[2]:start_indexes[3], :]
    threes_data = all_digits[start_indexes[3]:start_indexes[4], :]
    fours_data = all_digits[start_indexes[4]:start_indexes[5], :]
    fives_data = all_digits[start_indexes[5]:start_indexes[6], :]
    sixes_data = all_digits[start_indexes[6]:start_indexes[7], :]
    sevens_data = all_digits[start_indexes[7]:start_indexes[8], :]
    eights_data = all_digits[start_indexes[8]:start_indexes[9], :]
    nines_data = all_digits[start_indexes[9]:, :]

    return zeros_data, ones_data, twos_data, threes_data, fours_data, fives_data, sixes_data, sevens_data, eights_data, nines_data

#highest order routine
def get_first_fifty_images(inBinary = True, with_labels = False, zeroOnes = False):

    training_data = get_training_data(DATASET_PATH)
    if inBinary:
        training_data = make_binary(training_data, zeroOnes)
    zeros_data, ones_data, sixes_data, eights_data = get_subsets(training_data)
    
    if with_labels:
        return zeros_data[:50, :], ones_data[:50, :], sixes_data[:50, :], eights_data[:50, :]
    else:
        return zeros_data[:50, :-1], ones_data[:50, :-1], sixes_data[:50, :-1], eights_data[:50, :-1]

#highest order routine
def get_fifty_random_images(inBinary = True, with_labels = False):
    
    training_data = get_training_data(DATASET_PATH)
    if inBinary:
        training_data = make_binary(training_data)
    zeros, ones, sixes, eights = get_subsets(training_data)

    zero_idxs = np.random.randint(0, len(zeros), 200)
    one_idxs = np.random.randint(0, len(ones), 200)
    six_idxs = np.random.randint(0, len(sixes), 200)
    eight_idxs = np.random.randint(0, len(eights), 200)

    selected_zeros = np.take(zeros, zero_idxs[0:50], axis=0)
    selected_ones = np.take(ones, one_idxs[50:100], axis=0)
    selected_sixes = np.take(sixes, six_idxs[100:150], axis=0)
    selected_eights = np.take(eights, eight_idxs[150:200], axis=0)
    
    if with_labels:
        return selected_zeros, selected_ones, selected_sixes, selected_eights
    else: 
        return selected_zeros[:, :-1], selected_zeros[:, :-1], selected_sixes[:, :-1], selected_eights[:, :-1]
