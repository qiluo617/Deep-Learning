# Qi Luo
# A02274095

import numpy as np
w1 = np.array([[0.6, -0.7], [0.5, 0.4], [-0.6, 0.8]])
b1 = np.array([-0.4, -0.5])
w2 = np.array([[1], [1]])
b2 = np.array([-0.5])


def check_perc(array):
    for i in np.nditer(array, op_flags=['readwrite']):
        if i < 0:
            i[...] = 0
        else:
            i[...] = 1
    return array


print('prob4_3')
for x1 in [0, 1]:
    for x2 in [0, 1]:
        for x3 in [0, 1]:
            x = np.array([x1, x2, x3])
            l1 = check_perc(np.dot(x, w1) + b1)
            l2 = check_perc(np.dot(l1, w2) + b2)
            print(x, l2)

print('prob4_4')
for x1 in [0, 1]:
    for x2 in [0, 1]:
        for x3 in [0, 1]:
            x = np.array([x1, x2, x3])
            s1 = 1.0/(1.0+np.exp(-1 * (np.dot(x, w1) + b1)))
            s2 = 1.0/(1.0+np.exp(-1 * (np.dot(s1, w2) + b2)))
            print(x, s2)
