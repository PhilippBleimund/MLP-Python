from abc import ABC, abstractmethod

from .Optimizer import SGDOptimizer

import numpy as np


class _Normalizer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def normalize(self, x, inference) -> np.ndarray:
        pass

    @abstractmethod
    def train(self, grad_output):
        pass

    @abstractmethod
    def prepare_for_inference(self, *args, **kwargs):
        pass


class NoNormalizer(_Normalizer):
    def __init__(self):
        pass

    def normalize(self, x, inference) -> np.ndarray:
        return x

    def train(self, grad_output):
        return super().train(grad_output)

    def prepare_for_inference(self, *args, **kwargs):
        return super().prepare_for_inference(*args, **kwargs)


class BatchNormalizer(_Normalizer):
    def __init__(self, shape):
        self.gamma = np.ones(shape=shape)
        self.beta = np.zeros(shape=shape)
        self.epsilon = 1e-8

        self.optimizer_gamma = SGDOptimizer(shape=shape)
        self.optimizer_beta = SGDOptimizer(shape=shape)

        self.stable_mean = np.zeros(shape=shape)
        self.stable_variance = np.zeros(shape=shape)

        self.batch_counter = 0
        self.inference = False

    def _normalize_inference(self, x) -> np.ndarray:
        # epsilon is already in variance precomputed
        self.x_norm = (x - self.stable_mean) / np.sqrt(self.stable_variance)
        return self.gamma * self.x_norm + self.beta

    def clear_values(self):
        self.stable_mean = np.zeros_like(self.stable_mean)
        self.stable_variance = np.zeros_like(self.stable_variance)
        self.batch_counter = 0

    def prepare_for_inference(self, batch_size):
        self.stable_mean /= self.batch_counter
        self.stable_variance = (self.stable_variance / self.batch_counter) * \
            (batch_size/(batch_size-1))
        self.stable_variance += self.epsilon

        self.inference = True

    def normalize(self, x, inference=False) -> np.ndarray:
        if inference == True:
            return self._normalize_inference(x)

        batch_mean = np.mean(x, axis=0)
        batch_variance = np.var(x, axis=0)

        self.batch_counter += 1
        self.stable_mean += batch_mean
        self.stable_variance += batch_variance

        self.x_norm = (x - batch_mean)/np.sqrt(batch_variance + self.epsilon)

        y_out = self.gamma * self.x_norm + self.beta
        return y_out

    def train(self, grad_output):
        gradient_gamma = np.sum(grad_output * self.x_norm, axis=0)
        gradient_beta = np.sum(grad_output, axis=0)

        delta_gamma = self.optimizer_gamma(gradient_gamma)
        delta_beta = self.optimizer_beta(gradient_beta)

        self.gamma = np.add(self.gamma, delta_gamma)
        self.beta = np.add(self.beta, delta_beta)
