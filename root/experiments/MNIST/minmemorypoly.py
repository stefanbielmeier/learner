import numpy as np
from actualcap import get_recall_quality

def get_memorization_capacity(dataset):
    #Function that returns the minimum required capacity (polydegree) for memorization of a dataset of images
    #input is a 2D numpy array with #images x #pixels
    #output is the minimum required capacity (polydegree) for memorization of the dataset as an integer number
    
    #Steps to solve:
    #1. Calculate the recall qualities for (1 to 20) polydegrees for the dataset via get_recall_qualities
    #2. Stop calculating 
    #3. Find the polydegree in the recall quality array that has a value of 1
    #4. Return the polydegree
    
    estimated_polydegree = 1
    recall_quality = 0

    while recall_quality <= 0.99:
        estimated_polydegree = estimated_polydegree + 1
        recall_quality = get_recall_quality(dataset, estimated_polydegree, num_neurons=dataset.shape[1], plot_updated_images=False)
        print(recall_quality)
        print("increasing capacity")

    return estimated_polydegree



