import numpy as np
from abc import ABC, abstractmethod

from .activation_functions import get_activation_function, get_activation_function_abl

rng = np.random.default_rng(seed=1)


class _Layer(ABC):
    def __init__(self, size: int):
        self.size = size
        pass

    @abstractmethod
    def link_layer(self, prev_layer, next_layer):
        self.prev_layer = prev_layer
        self.next_layer = next_layer

    @abstractmethod
    def evaluate_layer(self):
        pass

    @abstractmethod
    def train_layer(self, *args, **kwargs):
        pass

    @abstractmethod
    def _get_output(self) -> np.ndarray:
        pass


class _ComputeLayer(_Layer):
    def __init__(self, num_perceptrons):
        super().__init__(num_perceptrons)
        self.b_values = np.zeros(shape=(num_perceptrons))
        self.o_values = np.zeros(shape=(num_perceptrons))
        self.errror_signals = np.zeros(shape=num_perceptrons)

        # Default Values for adam optimizer
        self.adam_alpha = 0.001
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_epsilon = 1e-8
        self.adam_time = 0

    def link_layer(self, prev_layer, next_layer):
        super().link_layer(prev_layer, next_layer)
        self.d_weights = np.zeros(shape=(self.size, prev_layer.size))
        self.weights = rng.normal(0, np.sqrt(2.0 / prev_layer.size),
                                  size=(self.size, prev_layer.size))

        self.adam_mt_old = np.zeros(shape=(self.size, prev_layer.size))
        self.adam_vt_old = np.zeros(shape=(self.size, prev_layer.size))

    @abstractmethod
    def _gradient_loss(self, *args, **kwargs) -> np.ndarray:
        pass

    def _adam_optimizer(self, delta):
        self.adam_time += 1
        m_t = self.adam_beta1 * self.adam_mt_old + (1-self.adam_beta1) * delta
        v_t = self.adam_beta2 * self.adam_vt_old + (1-self.adam_beta2) * np.square(delta)
        m_corrected = m_t / (1 - (self.adam_beta1 ** self.adam_time))
        v_corrected = v_t / (1 - (self.adam_beta2 ** self.adam_time))

        theta = self.adam_alpha * (m_corrected / (np.sqrt(v_corrected + self.adam_epsilon)))
        self.adam_mt_old = m_t
        self.adam_vt_old = v_t

        return theta


class InputLayer(_Layer):
    def __init__(self, input_size):
        super().__init__(input_size)
        self.o_values = np.zeros(shape=(input_size))

    def set_data(self, data):
        self.o_values = data

    def link_layer(self, prev_layer, next_layer):
        return super().link_layer(prev_layer, next_layer)

    def evaluate_layer(self):
        return super().evaluate_layer()

    def train_layer(self):
        return super().train_layer()

    def _get_output(self):
        return self.o_values


class PerceptronLayer(_ComputeLayer):
    def __init__(self, num_perceptrons, activation_method):
        super().__init__(num_perceptrons)
        self.activation_method = get_activation_function(activation_method)
        self.activation_method_abl = get_activation_function_abl(activation_method)

    def evaluate_layer(self):
        self.prev_layer.evaluate_layer()
        self.b_values = self.weights @ self.prev_layer._get_output()
        self.o_values = self.activation_method(self.b_values)

    def _gradient_loss(self) -> np.ndarray:
        self.errror_signals = (self.next_layer.weights.T @
                               self.next_layer.errror_signals) * self.activation_method(self.b_values)
        self.d_weights = np.outer(self.errror_signals, self.prev_layer.o_values)

        return self.d_weights

    def train_layer(self):
        delta = self._gradient_loss()
        self.weights = self.weights - self._adam_optimizer(delta)

        self.prev_layer.train_layer()

    def _get_output(self):
        return self.o_values


class PredictionLayer(_ComputeLayer):
    def __init__(self, num_perceptrons, classes: list):
        super().__init__(num_perceptrons)
        self.classes = classes
        self.activation_method = get_activation_function("softmax")

    def evaluate_layer(self):
        self.prev_layer.evaluate_layer()
        self.b_values = self.weights @ self.prev_layer._get_output()
        self.o_values = self.activation_method(self.b_values)

    def _gradient_loss(self, y_correct) -> np.ndarray:
        # loss function: cross entropy with softmax
        self.errror_signals = self.o_values - y_correct
        self.d_weights = np.outer(self.errror_signals, self.prev_layer.o_values)

        return self.d_weights

    def train_layer(self, correct_solution_idx=None, correct_solution=None):
        if correct_solution is not None:
            y_correct = correct_solution
        elif correct_solution_idx is not None:
            y_correct = np.zeros(shape=(len(self.classes)))
            y_correct[correct_solution_idx] = 1
        else:
            raise ValueError("At least one of the two has to be given")

        error = np.sum(0.5 * np.square(y_correct - self.o_values))
        cross_entropy = -np.sum(y_correct * np.log(np.clip(self.o_values, 1e-12, 1.-1e-12)))
        print(f"error: {error}")
        print(f"cross entropy: {cross_entropy}")

        delta = self._gradient_loss(y_correct)
        self.weights = self.weights - self._adam_optimizer(delta)

        self.prev_layer.train_layer()

    def _get_output(self):

        return self.classes[np.argmax(self.o_values)]
