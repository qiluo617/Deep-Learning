import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv

N = [15, 100]
sigma = [0, .05, .2]
degree = [1, 2, 9]

for n in N:
    for s in sigma:

        x = np.random.uniform(-1, 3, n)
        y = x*x - 3*x + 1 + np.random.normal(0, s, n)
        lab = "Sigma = " + str(s)
        plt.plot(x, y, 'r^', label=lab)

        for d in degree:
            x_d = (np.matrix([np.power(x, i) for i in range(0, d + 1)])).transpose()
            w = (pinv(x_d.transpose() * x_d).dot(x_d.transpose())).dot(y)
            y_s = x_d.dot(w.transpose())
            mse = np.power(y_s.reshape(n, ) - y, 2).sum()
            print("N=", n, "sigma=", s, "degree=", d)
            print("mse=", mse)
            print("weights:", w)
            txt = "Degree = " + str(d)
            plt.plot(x, y_s, 'o', label=txt)
        plt.legend(loc='upper right')
        plt.xlim(-2.0, 4.0)
        plt.ylim(-3.0, 6.0)
        plt.show()






