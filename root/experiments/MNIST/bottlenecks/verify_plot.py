import matplotlib.pyplot as plt
import numpy as np

x_axis = np.array([39, 6, 56, 36, 36, 36, 37, 23, 30, 38])
y_axis = [15, 53, 13, 17, 20, 13, 18, 21, 16, 19]

fitted = np.polyfit(x_axis, np.log(y_axis), 1)

ynew = np.exp(fitted[1]) * np.exp(fitted[0] * x_axis)

plt.scatter(x_axis, ynew, label="Hamming Distance vs. Memorization Capacity", c='blue')
plt.xlabel("HD")
plt.ylabel("Capacity")

plt.legend(loc='best')
plt.show()