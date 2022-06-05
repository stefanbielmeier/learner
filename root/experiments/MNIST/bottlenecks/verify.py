from root.experiments.MNIST.digits.subsets import DATASET_PATH, get_first_fifty_images, get_subsets, get_training_data


zeros, ones, twos, threes, fours, fives, sixes, sevens, eights, nines = get_subsets(get_training_data(DATASET_PATH))


