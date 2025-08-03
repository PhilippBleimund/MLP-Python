from .Optimizer import AdamOptimizer, SGDOptimizer
from .Normalization import NoNormalizer, BatchNormalizer
from .activation_functions import get_activation_function, get_activation_function_abl
import numpy as np
from abc import ABC, abstractmethod

# from line_profiler import LineProfiler
# lp = LineProfiler()


rng = np.random.default_rng(seed=1)


class _Layer(ABC):
    def __init__(self, size: int):
        self.size = size

        # for linter. The start and end of an model are None
        self.prev_layer: _Layer
        self.next_layer: _Layer

    @abstractmethod
    def prepare_for_training(self, optimizer, normalizer):
        pass

    def link_layer(self, prev_layer, next_layer):
        self.prev_layer = prev_layer
        self.next_layer = next_layer

    @abstractmethod
    def evaluate_layer(self, inference: bool) -> np.ndarray:
        pass

    @abstractmethod
    def train_layer(self, *args, **kwargs):
        pass

    def lock_layer(self):
        pass


class InputLayer(_Layer):
    def __init__(self, input_size):
        super().__init__(input_size)

        self.o_values: np.ndarray

    def set_data(self, data):
        if data.ndim == 1:
            self.o_values = data[np.newaxis, :]
        else:
            self.o_values = data

    def prepare_for_training(self, optimizer, normalizer):
        return super().prepare_for_training(optimizer, normalizer)

    def link_layer(self, prev_layer, next_layer):
        return super().link_layer(prev_layer, next_layer)

    def evaluate_layer(self, inference: bool):
        return self.o_values

    def train_layer(self, *args, **kwargs):
        return super().train_layer(*args, **kwargs)

    def _get_output(self):
        return self.o_values

    def lock_layer(self):
        return super().lock_layer()


class LinearLayer(_Layer):
    def __init__(self, size):
        super().__init__(size)

        self.o_values: np.ndarray
        self.error: np.ndarray
        self.weights: np.ndarray
        self.bias: np.ndarray
        self.d_weights: np.ndarray
        self.d_bias: np.ndarray

    def prepare_for_training(self, optimizer, normalizer):
        self.weights = rng.normal(0, np.sqrt(2.0 / self.prev_layer.size),
                                  size=(self.size, self.prev_layer.size))
        self.bias = np.zeros(shape=(self.size))

        if optimizer == "adam":
            self.weight_optimizer = AdamOptimizer(shape=(self.size, self.prev_layer.size))
            self.bias_optimizer = AdamOptimizer(shape=(self.size))
        elif optimizer == "sgd":
            self.weight_optimizer = SGDOptimizer(shape=(self.size, self.prev_layer.size))
            self.bias_optimizer = SGDOptimizer(shape=(self.size))

    def evaluate_layer(self, inference):
        o_values_prev = self.prev_layer.evaluate_layer(inference)
        self.o_values = np.dot(o_values_prev, self.weights.T) + self.bias

        return self.o_values

    def _calc_delta(self, prev_error_signal):
        self.d_weights = np.dot(
            prev_error_signal.T, self.prev_layer.o_values) / len(self.o_values)

        self.d_bias = np.mean(prev_error_signal, axis=0)

        return self.d_weights, self.d_bias

    def _gradient_loss(self, prev_error_signal):
        self.error = np.dot(prev_error_signal, self.weights)

        return self.error

    def train_layer(self, propagated_error):
        # apply own error to signal
        next_propagated_error = self._gradient_loss(propagated_error)

        delta_weight, delta_bias = self._calc_delta(propagated_error)

        delta_weight = self.weight_optimizer(delta_weight)
        delta_bias = self.bias_optimizer(delta_bias)

        self.weights = np.add(self.weights, delta_weight)
        self.bias = np.add(self.bias, delta_bias)

        self.prev_layer.train_layer(next_propagated_error)


class NormalizationLayer(_Layer):
    def __init__(self, size):
        super().__init__(size)

    def prepare_for_training(self, optimizer, normalizer):
        if normalizer == "batch":
            self.normalizer = BatchNormalizer(shape=self.size)
        else:
            self.normalizer = NoNormalizer()

    def evaluate_layer(self, inference: bool):
        o_values_prev = self.prev_layer.evaluate_layer(inference)
        self.o_values = self.normalizer(o_values_prev, inference)

        return self.o_values

    def _gradient_loss(self):
        pass


class ActivationLayer(_Layer):
    def __init__(self, size, activation_method):
        super().__init__(size)
        self.activation_method = get_activation_function(activation_method)
        self.activation_method_abl = get_activation_function_abl(activation_method)

        self.o_values: np.ndarray
        self.error: np.ndarray

    def prepare_for_training(self, optimizer, normalizer):
        return super().prepare_for_training(optimizer, normalizer)

    def evaluate_layer(self, inference):
        o_values_prev = self.prev_layer.evaluate_layer(inference)
        self.o_values = self.activation_method(o_values_prev)

        return self.o_values

    def _gradient_loss(self, prev_output_error):
        self.error = np.multiply(prev_output_error, self.activation_method_abl(
            self.prev_layer.o_values))

        return self.error

    def train_layer(self, propagated_error):
        # nothing to train

        # apply own error
        propagated_error = self._gradient_loss(propagated_error)

        self.prev_layer.train_layer(propagated_error)


class PredictionLayer(_Layer):
    """
    Expects an Linear Layer with the same size as previous layer.
    """

    def __init__(self, size, classes):
        super().__init__(size)
        self.classes = classes
        self.activation_method = get_activation_function("softmax")

        self.o_values: np.ndarray
        self.error: np.ndarray

    def prepare_for_training(self, optimizer, normalizer):
        return super().prepare_for_training(optimizer, normalizer)

    def evaluate_layer(self, inference: bool):
        o_values_prev = self.prev_layer.evaluate_layer(inference)
        self.o_values = self.activation_method(o_values_prev)

        return self.o_values

    def _gradient_loss(self, out_correct):
        self.error = np.subtract(self.o_values, out_correct)

        return self.error

    def train_layer(self, correct_solution_idx=None, correct_solution=None):
        if correct_solution is not None:
            y_correct = correct_solution
        elif correct_solution_idx is not None:
            y_correct = np.zeros(shape=(len(correct_solution_idx), len(self.classes)))
            for i, idx in enumerate(correct_solution_idx):
                y_correct[i, idx] = 1
        else:
            raise ValueError("At least one of the two has to be given")

        # create error signal that propagates back
        propagated_error = self._gradient_loss(y_correct)

        self.prev_layer.train_layer(propagated_error)

    def _get_output(self):
        a = np.argmax(self.o_values, axis=1)
        return [i for i in a]

    def get_prediction(self):
        a = np.argmax(self.o_values, axis=1)
        return [self.classes[i] for i in a]
