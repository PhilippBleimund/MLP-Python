from abc import ABC, abstractmethod

from .Optimizer import AdamOptimizer, SGDOptimizer

import numpy as np


class _Normalizer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def normalize(self, x, inference) -> np.ndarray:
        pass

    @abstractmethod
    def gradient(self, x, input_gradient):
        pass

    @abstractmethod
    def train(self, input_gradient):
        pass

    def __call__(self, *args, **kwargs):
        return self.normalize(*args, **kwargs)


class NoNormalizer(_Normalizer):
    def __init__(self):
        pass

    def normalize(self, x, inference) -> np.ndarray:
        return x

    def gradient(self, x, input_gradient):
        return input_gradient

    def train(self, input_gradient):
        return super().train(input_gradient)


class BatchNormalizer(_Normalizer):
    def __init__(self, shape, optimizer):
        self.gamma = np.ones(shape=shape)
        self.beta = np.zeros(shape=shape)
        self.epsilon = 1e-8
        self.momentum = 0.1

        self.stable_mean = np.zeros(shape=shape)
        self.stable_variance = np.zeros(shape=shape)

        if optimizer == "adam":
            self.optimizer_gamma = AdamOptimizer(shape=shape)
            self.optimizer_beta = AdamOptimizer(shape=shape)
        elif optimizer == "sgd":
            self.optimizer_gamma = SGDOptimizer(shape=shape)
            self.optimizer_beta = SGDOptimizer(shape=shape)

    def _normalize_inference(self, x) -> np.ndarray:
        # epsilon is already in variance precomputed
        self.x_norm = (x - self.stable_mean) / np.sqrt(self.stable_variance)
        return self.gamma * self.x_norm + self.beta

    def normalize(self, x, inference=False) -> np.ndarray:
        if inference == True:
            return self._normalize_inference(x)

        batch_mean = np.mean(x, axis=0)
        batch_variance = np.var(x, axis=0)

        self.stable_mean = (1 - self.momentum) * self.stable_mean + self.momentum * batch_mean
        self.stable_variance = (1 - self.momentum) * self.stable_variance + \
            self.momentum * batch_variance

        self.x_norm = (x - batch_mean)/np.sqrt(batch_variance + self.epsilon)

        y_out = self.gamma * self.x_norm + self.beta
        return y_out

    def train(self, input_gradient):
        gradient_gamma = np.sum(input_gradient * self.x_norm, axis=0)
        gradient_beta = np.sum(input_gradient, axis=0)

        delta_gamma = self.optimizer_gamma(gradient_gamma)
        delta_beta = self.optimizer_beta(gradient_beta)

        self.gamma = np.add(self.gamma, delta_gamma)
        self.beta = np.add(self.beta, delta_beta)

    def gradient(self, x, input_gradient):
        batch_mean = np.mean(x, axis=0)
        batch_variance = np.var(x, axis=0)

        error_x_norm = input_gradient * self.gamma
        inverse_sqrt = 1/np.sqrt(batch_variance + self.epsilon)
        error_variance = np.sum(error_x_norm * (x - batch_mean), axis=0) * \
            0.5 * np.power(inverse_sqrt, 3)
        error_mean = np.sum(error_x_norm * -inverse_sqrt, axis=0) + \
            error_variance * np.sum(-2*(x - batch_mean), axis=0) / len(x)
        error_x = error_x_norm * inverse_sqrt + error_variance * \
            (2 * (x - batch_mean)) / len(x) + error_mean / len(x)

        return error_x
