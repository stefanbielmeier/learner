import matplotlib.pyplot as plt
import numpy as np

min_HD = np.array([39, 6, 56, 36, 36, 36, 37, 23, 30, 38])
mean_HD = np.array([123, 62, 131, 116, 104, 118, 118, 101, 122, 93])

y_axis = [15, 53, 13, 17, 20, 13, 18, 21, 16, 19]

min_set = np.vstack((min_HD, y_axis))
min_set = min_set[:, min_set[0, :].argsort()]

mean_set = np.vstack((mean_HD, y_axis))
mean_set = mean_set[:, mean_set[0, :].argsort()]

print(min_set)
print(mean_set)

fitted_min = np.polyfit(min_set[0,:], np.log(min_set[1,:]), 1)
fitted_mean = np.polyfit(mean_set[0,:], np.log(mean_set[1,:]), 1)

ynew_min = np.exp(fitted_min[1]) * np.exp(fitted_min[0] * min_set[0,:])
ynew_mean = np.exp(fitted_mean[1]) * np.exp(fitted_mean[0] * mean_set[0,:])

#plt.scatter(mean_set[0,:], mean_set[1,:], label="MEAN Hamming Distance vs. Memorization Capacity", c='blue')
plt.scatter(min_set[0,:], min_set[1,:], label="MIN Hamming Distance vs. Memorization Capacity", c='red')

plt.plot(min_set[0,:], ynew_min, label="Fitted line", c='red')
#plt.plot(mean_set[0,:], ynew_mean, label="Fitted line", c='blue')

plt.xlabel("HD")
plt.ylabel("Capacity")

plt.legend(loc='upper right')
plt.show()