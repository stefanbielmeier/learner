from root.experiments.MNIST.bottlenecks.verify.exponential.create_subsets import create_subset
from root.experiments.MNIST.digits.subsets import DATASET_PATH, get_subsets, get_training_data, make_binary

training_data = get_training_data(DATASET_PATH)
training_data = make_binary(training_data, zeroOnes = False)
_, ones, _, _, _, _, _, _, _, _ = get_subsets(training_data)
onehundred = create_subset(ones, 50, 40)