import numpy as np


def get_activation_function(activation_function):
    if activation_function == "sigmoid":
        return lambda z: sigmoid(z)
    elif activation_function == "softmax":
        return lambda z: softmax(z)
    elif activation_function == "relu":
        return lambda z: relu(z)
    return lambda z: z


def get_activation_function_abl(activation_function):
    if activation_function == "sigmoid":
        return lambda z: sigmoid_abl(z)
    elif activation_function == "softmax":
        return lambda z: 1
    elif activation_function == "relu":
        return lambda z: relu_abl(z)

    return lambda z: z


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def sigmoid_abl(z):
    return sigmoid(z) * (1 - sigmoid(z))


def relu(z):
    return np.maximum(z, 0)


def relu_abl(z):
    return 1. * (z > 0)


def softmax(z):
    shiftx = z - np.max(z)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)
