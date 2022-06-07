import numpy as np
from root.experiments.MNIST.digits.subsets import DATASET_PATH, get_first_fifty_images, get_subsets, get_training_data, make_binary
from root.experiments.MNIST.information.hamming import get_bottleneck_idxs, hamming_distance, mean_hamming_distance, min_hamming_distance


training_data = get_training_data(DATASET_PATH)
training_data = make_binary(training_data, zeroOnes = False)
_, ones, _, _, fours, _, _, _, _, _ = get_subsets(training_data)

#bottleneck0s = get_bottleneck_idxs(first_500)[0]
#hd0_mean = mean_hamming_distance(first_500)
#hd0 = min_hamming_distance(first_500)

#print(bottleneck0s)
#print(hd0_mean) #60
#print(hd0) #2

#TODO: Basically: create 80 datasets, and check what happens.

def create_subset(full_dataset, num_memories, goal_hamming_distance):
    """
    Return a subset.
    Pick pairs of memories from a dataset until num_memories is reached.
    No pair of memories in the subset can have a hamming distance
    smaller than min_hamming_distance.
    """
    subset = []
    added_idxs = []

    while len(subset) < num_memories - 1:
        idx1 = np.random.randint(0, full_dataset.shape[0])
        #no duplications
        if idx1 not in added_idxs:
                # check if the hamming distance between randomly picked memory and all other memories in subset is too low
                new_memory = full_dataset[idx1,:]
                fitting_hd = True
                subset_idx = 0
                # if the hamming distance is too low, don't add the memory
                while fitting_hd and subset_idx < len(subset):
                    if hamming_distance(subset[subset_idx], new_memory) < goal_hamming_distance:
                        fitting_hd = False
                    subset_idx += 1
                    
                if fitting_hd:
                    subset.append(new_memory)
                    added_idxs.append(idx1)
    """
    #ensure the last memory has exactly the right hamming distance to all
    while len(subset) < num_memories:
        for idx1 in range(full_dataset.shape[0]):
            if idx1 not in added_idxs:
                new_memory = full_dataset[idx1,:]
                found_elem = False
                subset_idx = 0
                    # if the hamming distance is too low, don't add the memory
                while not(found_elem) and subset_idx < len(subset):
                    if hamming_distance(subset[subset_idx], new_memory) == goal_hamming_distance:
                        fitting_hd = True
                    subset_idx += 1
                        
                if found_elem:
                    subset.append(new_memory)
                    added_idxs.append(idx1)

    """
    return np.array(subset, dtype=np.float64)

def create_subsets(full_dataset, num_memories, hd_range = []):
    subsets = []
    for hd in hd_range:
        subsets.append(create_subset(full_dataset, num_memories, hd))
        print("new subset appended: ", hd)
    return subsets

subsets = create_subsets(fours, 50, hd_range=range(1,100,2))
with open('subsets.npy', 'wb') as f:
    np.save(f, subsets)
