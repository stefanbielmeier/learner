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

def get_mean_hds(subsets):
    mean_hds = []
    for subset in subsets:
        mean_hds.append(mean_hamming_distance(subset))
    return mean_hds

def load_subsets(filepath):
    array = []
    with open(filepath, 'rb') as f:
        array = np.load(f)
    return array

def test_subsets(subsets):
    mem_caps = []
    for subset in subsets:
        bottleneck_idxs = get_bottleneck_idxs(subset)[0]
        mem_cap = get_memorization_capacity(subset, recall_quality = 1.0, startAt = 6, verbose=True, test_idxs = bottleneck_idxs) 
        mem_caps.append(mem_cap)

    return mem_caps
    
def main():
    subsets_40_120 = load_subsets("subsets_40_120.npy")
    subsets_10_22 = load_subsets("subsets_10_22.npy")
    mem_caps1 = test_subsets(subsets_40_120)
    mem_caps2 = test_subsets(subsets_10_22)
    with open('mem_caps_subsets_40_120.npy', 'wb') as f:
       np.save(f, mem_caps1)
    with open('mem_caps_subsets_10_22.npy', 'wb') as f:
       np.save(f, mem_caps2)

if __name__ == "__main__":
    main()