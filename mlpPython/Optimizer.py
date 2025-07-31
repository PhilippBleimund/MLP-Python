from abc import ABC, abstractmethod

import numpy as np


class _Optimizer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def create_delta_weights(self, *args, **kwargs) -> np.ndarray:
        pass


class AdamOptimizer(_Optimizer):
    def __init__(self, shape):
        # Default Values for adam optimizer
        self.adam_alpha = 0.001
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_epsilon = 1e-8
        self.adam_time = 0

        self.adam_mt_old = np.zeros(shape=shape)
        self.adam_vt_old = np.zeros(shape=shape)

    def _adam_optimizer(self, adam_g, adam_mt, adam_vt):
        m_t = self.adam_beta1 * adam_mt + (1-self.adam_beta1) * adam_g
        np.square(adam_g, out=adam_g)
        v_t = self.adam_beta2 * adam_vt + (1-self.adam_beta2) * adam_g
        adam_beta1_power = self.adam_beta1 ** self.adam_time
        adam_beta2_power = self.adam_beta2 ** self.adam_time
        m_corrected = m_t / (1 - adam_beta1_power)
        v_corrected = v_t / (1 - adam_beta2_power)

        theta = - self.adam_alpha * (m_corrected / (np.sqrt(v_corrected + self.adam_epsilon)))

        return m_t, v_t, theta

    def create_delta_weights(self, gradient) -> np.ndarray:
        self.adam_time += 1

        self.adam_mt_old, self.adam_vt_old, theta = self._adam_optimizer(
            gradient, self.adam_mt_old, self.adam_vt_old)

        return theta


class SGDOptimizer(_Optimizer):
    def __init__(self, shape, learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=False):
        self.learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.nesterov = nesterov
        if momentum > 0:
            self.velocity = np.zeros(shape=shape)

    def create_delta_weights(self, gradient) -> np.ndarray:
        if self.momentum == 0.0:
            return - self.learning_rate * gradient

        self.velocity = self.momentum * self.velocity - self.learning_rate * gradient

        # doesnt work but should be mathematiclt correct
        if self.nesterov == True:
            return self.momentum * self.velocity - self.learning_rate * gradient
        else:
            return self.velocity
