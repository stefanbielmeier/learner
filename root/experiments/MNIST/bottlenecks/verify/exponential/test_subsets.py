import numpy as np
from root.experiments.MNIST.bottlenecks.verify.plot import make_sorted_set, plot_scatter_with_fitted_line
from root.experiments.MNIST.information.hamming import get_bottleneck_idxs, mean_hamming_distance, min_hamming_distance

from root.utils import record_stdout
from root.experiments.MNIST.capacity.calcmemcap import get_memorization_capacity


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
        mem_cap = get_memorization_capacity(subset, recall_quality = 1.0, startAt = 12, verbose=True, test_idxs = bottleneck_idxs) 
        mem_caps.append(mem_cap)

        break

    return mem_caps, hds
    
def main():
    subsets = load_subsets("subsets.npy")
    mem_caps, hds = test_subsets(subsets, min_hd = False)
    dataset = make_sorted_set(mem_caps, hds)
    plot_scatter_with_fitted_line(dataset, "hd", "mem_cap", "Plot min_hd vs. mem_cap", color="blue")
    with "test_subsets.npy" as f:
        np.save(f, dataset)

if __name__ == "__main__":
    main()