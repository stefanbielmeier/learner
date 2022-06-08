from root.experiments.MNIST.bottlenecks.verify.exponential.test_subsets import get_mean_hds, get_min_hds, load_subsets
from root.experiments.MNIST.bottlenecks.verify.plot import plot_scatter_with_fitted_line

import numpy as np

def main():
    #create_subsets() #do when needed
    subsets = load_subsets("all_subsets.npy")
    memcaps = load_subsets("memcaps.npy")
    print(memcaps)
    hds = get_mean_hds(subsets)
    dataset = np.vstack((hds, memcaps))

    plot_scatter_with_fitted_line(dataset, "memcaps.png", "Mean HD", "Mem Cap", "red")

if __name__ == "__main__":
    main()