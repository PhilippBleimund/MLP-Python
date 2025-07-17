import numpy as np
from abc import ABC, abstractmethod

from activation_functions import get_activation_function

rng = np.random.default_rng(seed=1)


class Layer(ABC):
    def __init__(self, size: int, prev_layer, next_layer):
        self.size = size
        self.prev_layer = prev_layer
        self.next_layer = next_layer
        pass

    @abstractmethod
    def evaluate_layer(self):
        pass

    @abstractmethod
    def _get_output(self) -> np.ndarray:
        pass


class InputLayer(Layer):
    def __init__(self, input_size):
        super().__init__(input_size, None)
        self.o_values = np.zeros(shape=(input_size))

    def set_data(self, data):
        self.o_values = data

    def evaluate_layer(self):
        return super().evaluate_layer()

    def _get_output(self):
        return self.o_values


class PerceptronLayer(Layer):
    def __init__(self, num_perceptrons, activation_method, prev_layer: Layer):
        super().__init__(num_perceptrons, prev_layer)
        self.activation_method = get_activation_function(activation_method)
        self.weights = rng.normal(0, 0.1, size=(num_perceptrons, prev_layer.size))
        self.b_values = np.zeros(shape=(num_perceptrons))
        self.o_values = np.zeros(shape=(num_perceptrons))

    def evaluate_layer(self):
        self.prev_layer.evaluate_layer()
        self.b_values = self.weights @ self.prev_layer._get_output()

    def _get_output(self):
        return self.o_values


class PredictionLayer(Layer):
    def __init__(self, classes: list, prev_layer: Layer):
        super().__init__(len(classes), prev_layer)
        self.classes = classes

    def evaluate_layer(self):
        self.prev_layer.evaluate_layer()

    def _get_output(self):
        prev_layer_results = self.prev_layer._get_output()

        return self.classes[np.argmax(prev_layer_results)]
