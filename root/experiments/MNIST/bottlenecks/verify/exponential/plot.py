from root.experiments.MNIST.bottlenecks.verify.exponential.test_subsets import load_subsets


def main():
    #create_subsets() #do when needed
    memcaps = load_subsets("memcaps.npy")
    print(memcaps)
    

if __name__ == "__main__":
    main()