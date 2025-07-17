import numpy as np


def get_activation_function(activation_function):
    if activation_function == "sigmoid":
        return lambda z: sigmoid(z)
    elif activation_function == "softmax":
        return lambda z: softmax(z)


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def softmax(z):
    return np.exp(z-np.max(z))/np.sum(z)
