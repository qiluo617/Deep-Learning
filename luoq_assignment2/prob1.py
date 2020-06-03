# Qi Luo
# A02274095

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv

N = [15, 100]
sigma = [.05]
degree = [9]
lamb = [0, 0.5, 5]

for i in range(0, len(N)):
    for s in sigma:

        x = np.linspace(-1, 3, N[i])
        y = x*x - 3*x + 1 + np.random.normal(0, s, N[i])
        lab = "Original = " + str(N[i])
        plt.subplot(1, 2, i+1)
        plt.plot(x, y, '--', label=lab)

        for d in degree:
            for l in lamb:
                x_d = (np.matrix([np.power(x, i) for i in range(0, d + 1)])).transpose()
                w = (pinv(l * np.identity(10) + x_d.transpose() * x_d).dot(x_d.transpose())).dot(y)
                y_s = x_d.dot(w.transpose())
                mse = np.power(y_s.reshape(N[i], ) - y, 2).sum()
                print("N=", N[i], "lamb=", l)
                print("mse=", mse)
                print("weights:", w)
                txt = "lambda = " + str(l)
                plt.plot(x, y_s, '--', label=txt)
        plt.legend(loc='upper right')
        plt.xlim(-2.0, 4.0)
        plt.ylim(-3.0, 6.0)
    print('')
plt.show()
