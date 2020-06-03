

import numpy as np


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


x1 = np.array([[0.05], [0.1]])
w1 = np.array([[0.15, 0.2], [0.25, 0.3]])
b1 = np.array([[0.35], [0.35]])

h = sigmoid(np.dot(w1, x1) + b1)


w2 = np.array([[0.4, 0.45], [0.5, 0.55]])
b2 = np.array([[0.6], [0.6]])

o = sigmoid(np.dot(w2, h) + b2)

out = np.array([[0.01], [0.99]])
output_err = o - out

delta_1 = np.multiply(np.dot(np.transpose(w2), output_err), sigmoid_prime(np.dot(w1, x1) + b1))

print(delta_1)
print(output_err)

weight_layer1 = np.dot(x1, np.transpose(delta_1))
print(weight_layer1)

weight_layer2 = np.dot(h, np.transpose(output_err))
print(weight_layer2)


