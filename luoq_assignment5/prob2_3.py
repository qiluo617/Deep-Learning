import numpy as np
import matplotlib.pyplot as plt
import math


def f(x):
    return 2 / x * math.log(x / 2 * (1 + math.sqrt(1 - 4 / x)) - 1)


X = np.arange(4.0, 14.0, 0.05)
Y = np.array([f(x) for x in X])

plt.plot(X, Y)
plt.plot(6.9, 0.45, 'ro')
plt.show()