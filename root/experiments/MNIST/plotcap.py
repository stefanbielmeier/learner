import numpy as np
import matplotlib.pyplot as plt


x = np.array([0.02, 0.04, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])*100
y = [2, 4, 10, 11, 17, 20, 20, 24, 23, 23, 27, 29]

y2 = [2,4,10,20,30,40,50,60,70,80,90,100]

plt.plot(x, y, label="capacity progression for % of dataset")
plt.plot(x, y2, label="linear")
plt.xlabel("Percent of 100 images used as memories")
plt.ylabel("Capacity required to memorize (in terms of n)")
plt.yscale('linear')

plt.legend(loc='best')
plt.show()