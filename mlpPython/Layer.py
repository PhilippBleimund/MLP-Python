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


class PerceptronLayer(_Layer):
    def __init__(self, num_perceptrons, activation_method):
        super().__init__(num_perceptrons)
        self.activation_method = get_activation_function(activation_method)
        self.activation_method_abl = get_activation_function_abl(activation_method)
        self.b_values = np.zeros(shape=(num_perceptrons))
        self.o_values = np.zeros(shape=(num_perceptrons))
        self.errror_signals = np.zeros(shape=num_perceptrons)

    def link_layer(self, prev_layer, next_layer):
        super().link_layer(prev_layer, next_layer)
        self.d_weights = np.zeros(shape=(self.size, prev_layer.size))
        self.weights = rng.normal(0, 0.1, size=(self.size, prev_layer.size))

    def evaluate_layer(self):
        self.prev_layer.evaluate_layer()
        self.b_values = self.weights @ self.prev_layer._get_output()
        self.o_values = self.activation_method(self.b_values)

    def train_layer(self, learning_rate):
        for i in range(self.size):
            self.errror_signals[i] = np.sum(self.next_layer.errror_signals * self.next_layer.weights[:, i]) * \
                self.activation_method_abl(self.b_values[i])
            for j in range(self.prev_layer.size):
                self.d_weights[i, j] = self.errror_signals[i] * self.prev_layer.o_values[j]

        self.weights += -learning_rate * self.d_weights

    def _get_output(self):
        return self.o_values


class PredictionLayer(_Layer):
    def __init__(self, num_perceptrons, activation_method, classes: list):
        super().__init__(num_perceptrons)
        self.activation_method = get_activation_function(activation_method)
        self.activation_method_abl = get_activation_function_abl(activation_method)
        self.b_values = np.zeros(shape=(num_perceptrons))
        self.o_values = np.zeros(shape=(num_perceptrons))
        self.errror_signals = np.zeros(shape=num_perceptrons)
        self.classes = classes
        self.cross_entropy_loss = 1e10
        self.learning_rate = 1e-4

    def link_layer(self, prev_layer, next_layer):
        super().link_layer(prev_layer, next_layer)
        self.d_weights = np.zeros(shape=(self.size, prev_layer.size))
        self.weights = rng.normal(0, 0.1, size=(self.size, prev_layer.size))

    def evaluate_layer(self):
        self.prev_layer.evaluate_layer()
        self.b_values = self.weights @ self.prev_layer._get_output()
        self.o_values = self.activation_method(self.b_values)

    def train_layer(self, correct_solution_idx=None, correct_solution=None):
        if correct_solution is not None:
            y_correct = correct_solution
        elif correct_solution_idx is not None:
            y_correct = np.zeros(shape=(len(self.classes)))
            y_correct[correct_solution_idx] = 1
        else:
            raise ValueError("At least one of the two has to be given")

        for i in range(self.size):
            self.errror_signals[i] = -(y_correct[i] - self.o_values[i]) * \
                self.activation_method_abl(self.b_values[i])
            for j in range(self.prev_layer.size):
                self.d_weights[i, j] = self.errror_signals[i] * self.prev_layer.o_values[j]

        epsilon = 1e-12  # small value to avoid log(0)
        y_pred = np.clip(self.o_values, epsilon, 1. - epsilon)
        cross_entropy_loss = - np.sum(y_correct * np.log(y_pred))

        print("prev cross: " + str(self.cross_entropy_loss))
        print("now cross: " + str(cross_entropy_loss))
        if cross_entropy_loss < self.cross_entropy_loss:
            self.learning_rate = min(self.learning_rate + 1e-3, 0.01)
        elif abs(cross_entropy_loss - self.cross_entropy_loss) > 0.1:
            self.learning_rate = max(self.learning_rate - 1e-3, 1e-6)
        self.cross_entropy_loss = cross_entropy_loss
        print("learning_rate: " + str(self.learning_rate))

        self.weights += -self.learning_rate * self.d_weights

        self.prev_layer.train_layer(self.learning_rate)

    def _get_output(self):

        return self.classes[np.argmax(self.o_values)]
