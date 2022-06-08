import numpy as np
from root.experiments.MNIST.bottlenecks.verify.plot import make_sorted_set, plot_scatter_with_fitted_line
from root.experiments.MNIST.information.hamming import get_bottleneck_idxs, mean_hamming_distance, min_hamming_distance

from root.utils import record_stdout
from root.experiments.MNIST.capacity.calcmemcap import get_memorization_capacity

def get_min_hds(subsets):
    min_hds = []
    for subset in subsets:
        min_hds.append(min_hamming_distance(subset))
    return min_hds

def load_subsets(filepath):
    array = []
    with open(filepath, 'rb') as f:
        array = np.load(f)
    return array

def test_subsets(subsets, min_hd = True):
    mem_caps = []
    hds = []
    for subset in subsets:
        if min_hd:
            hd = min_hamming_distance(subset)
        else:
            hd = mean_hamming_distance(subset)
        hds.append(hd)

        bottleneck_idxs = get_bottleneck_idxs(subset)[0]
        mem_cap = get_memorization_capacity(subset, recall_quality = 1.0, startAt = 6, verbose=True, test_idxs = bottleneck_idxs) 
        mem_caps.append(mem_cap)

    return hds, mem_caps
    
def main():
    subsets = load_subsets("subsets2.npy")
    #hds, mem_caps = test_subsets(subsets, min_hd = False)
    #dataset = make_sorted_set(hds, mem_caps)    
    dataset = load_subsets("test_subsets.npy")
    dataset = np.flip(dataset, axis=0)
    min_hds = sorted(get_min_hds(subsets), reverse=True)
    #dataset[0,:] = min_hds
    print(dataset)
    plot_scatter_with_fitted_line(dataset, "Plot min_hd vs. mem_cap", "hd", "mem_cap", color="blue")
    #with open('test_subsets.npy', 'wb') as f:
    #   np.save(f, dataset)

if __name__ == "__main__":
    main()