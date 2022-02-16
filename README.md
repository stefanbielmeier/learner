# learner
Make unsupervised learning faster

## Capacity progression for Hopfield Networks
The first question I am trying to answer here: How can I estimate the capacity that a Hopfield Network needs to memorize a certain dataset? 
The second question I am trying to answer here: How can I estimate the capacity that a Hopfield Network needs to generalize given a certain dataset? 

**Memorization and Generalization**
Memorization here means 100% accuracy in reproduction of an image. Stable states for each memory in the network.
Generalization means that the Hopfield network learns some sort of rule that is applicable to data beyond the initial training data. For example, the clustering of previously unknown images into the learned categories.

**Potential Implications**
With a memorization capacity estimate for a dataset, I would be able to determine an architecture and hyperparameters for a Hopfield network that memorizes the given dataset _before training_ it. Similarly, a generalization capacity estimate would allow us to determine the architecture and hyperparameters of a Hopfield network that generalizes given the dataset _before training_ it. The latter obviously also depends whether there are generalizable patterns in the data in the first place.

If I compare this approach to other recent AutoML approaches. The current ones heavily involve _search_ for hyperparameters and architectures to achieve the desired results. In the most efficient case, the search requires at least one full training run of the Hopfield Network. In most cases, the median search probably involves many iterations. And thus, current approaches require many partial and full training runs which costs time and compute power (meaning: money). 

So, if I am successful, a lot less training runs and compute power are needed to build a satisfactory model for whatever task needs to be achieved. Basically, this could improve the speed of unsupervised AutoML by quite a bit. Potentially, an order of magnitude. But that is still to be found out.
