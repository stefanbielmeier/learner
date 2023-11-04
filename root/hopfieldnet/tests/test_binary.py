import numpy as np
from root.hopfieldnet.densehopfield import HopfieldNetwork

from root.utils import plot_img


def main():

    T = np.array([[1, 1, 1, 1, 1], [-1, -1, 1, -1, -1], [-1, -1,
                 1, -1, -1], [-1, -1, 1, -1, -1], [-1, -1, 1, -1, -1]])
    H = np.array([[1, -1, -1, -1, 1], [1, -1, -1, -1, 1],
                 [1, 1, 1, 1, 1], [1, -1, -1, -1, 1], [1, -1, -1, -1, 1]])
    E = np.array([[1, 1, 1, 1, 1], [1, -1, -1, -1, -1, ],
                 [1, 1, 1, 1, 1], [1, -1, -1, -1, -1], [1, 1, 1, 1, 1]])

    S = np.array([[1, 1, 1, 1, 1], [1, -1, -1, -1, -1, ],
                 [1, 1, 1, 1, 1], [-1, -1, -1, -1, 1], [1, 1, 1, 1, 1]])

    # ten noisy bits
    noisy_t = np.array([[-1, -1, 1, 1, 1], [-1, -1, 1, -1, 1],
                       [-1, 1, 1, -1, -1], [-1, 1, 1, -1, 1], [1, -1, 1, 1, 1]])

    # three noise bits can be restored with two or three memories – no problem
    noisy_h = np.array([[-1, -1, -1, -1, 1], [1, 1, -1, -1, 1],
                       [1, 1, 1, 1, 1], [1, -1, -1, -1, 1], [1, 1, 1, -1, 1]])

    # six noise bits and little overlap => Corrupted H
    noisy_e = np.array([[1, 1, -1, 1, 1], [-1, 1, -1, 1, -1],
                       [1, 1, 1, 1, 1], [-1, -1, -1, -1, -1], [1, 1, -1, 1, 1]])

    # adding a fourth memory makes the network fail and get really random / noisy results!
    X = np.array([[1, -1, -1, -1, 1], [-1, 1, -1, 1, -1],
                 [-1, -1, 1, -1, -1], [-1, 1, -1, 1, -1], [1, -1, -1, -1, 1]])
    # six noise bits
    noisy_x = np.array([[-1, -1, -1, -1, 1], [-1, -1, -1, 1, -1],
                       [1, 1, 1, -1, 1], [1, 1, -1, 1, -1], [1, -1, -1, 1, -1]])

    fourmems = np.stack([T, H, E, X], axis=0)
    # flattens all except first dim, => 2D matrix
    fourmems = fourmems.reshape(fourmems.shape[0], -1)

    newnet = HopfieldNetwork(25, 10)
    newnet.learn(fourmems)

    noisy_s = np.array([[1, 1, 1, -1, 1], [-1, 1, -1, -1, -1],
                       [1, 1, 1, 1, 1], [-1, -1, 1, -1, 1], [1, 1, 1, 1, 1]])

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

    """
    print('S')
    plot_img(noisy_s, 5)
    newnet.update(noisy_s.flatten())
    newnet.plot()
    """


if __name__ == "__main__":
    main()
