from .Optimizer import AdamOptimizer, SGDOptimizer
from .Normalization import NoNormalizer, BatchNormalizer
from .activation_functions import get_activation_function, get_activation_function_abl
import numpy as np
from abc import ABC, abstractmethod
from typing import Any

# from line_profiler import LineProfiler
# lp = LineProfiler()


rng = np.random.default_rng(seed=1)


class _Layer(ABC):
    def __init__(self, size: int):
        self.size = size
        pass

    @abstractmethod
    def prepare_for_training(self, max_batch_size, optimizer, normalizer):
        self.max_batch_size = max_batch_size

    @abstractmethod
    def link_layer(self, prev_layer, next_layer):
        self.prev_layer = prev_layer
        self.next_layer = next_layer

    @abstractmethod
    def evaluate_layer(self, input_batch_size, inference: bool):
        pass

    @abstractmethod
    def train_layer(self, *args, **kwargs):
        pass

    @abstractmethod
    def _get_output(self) -> Any:
        pass

    @abstractmethod
    def lock_layer(self):
        pass


class _ComputeLayer(_Layer):
    def __init__(self, num_perceptrons):
        super().__init__(num_perceptrons)

        # Default Values for adam optimizer
        self.adam_alpha = 0.001
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_epsilon = 1e-8
        self.adam_time = 0

    def prepare_for_training(self, max_batch_size, optimizer, normalizer):
        super().prepare_for_training(max_batch_size, optimizer, normalizer)
        # after summing
        self.b_values = np.zeros(shape=(max_batch_size, self.size))
        # after activation
        self.o_values = np.zeros(shape=(max_batch_size, self.size))

        # error signals for batch and weight gradient is mean of batch
        self.output_error = np.zeros(shape=(max_batch_size, self.size))
        self.error_signals = np.zeros(shape=(max_batch_size, self.size))
        self.d_weights = np.zeros(shape=(self.size, self.prev_layer.size))
        self.weights = rng.normal(0, np.sqrt(2.0 / self.prev_layer.size),
                                  size=(self.size, self.prev_layer.size))
        self.bias = np.zeros(shape=(self.size))

        if optimizer == "adam":
            self.weight_optimizer = AdamOptimizer(shape=(self.size, self.prev_layer.size))
            self.bias_optimizer = AdamOptimizer(shape=(self.size))
        elif optimizer == "sgd":
            self.weight_optimizer = SGDOptimizer(shape=(self.size, self.prev_layer.size))
            self.bias_optimizer = SGDOptimizer(shape=(self.size))

        if normalizer == "no":
            self.normalizer = NoNormalizer()
        elif normalizer == "batch":
            self.normalizer = BatchNormalizer(shape=self.size)

    def lock_layer(self):
        self.normalizer.prepare_for_inference(self.max_batch_size)

    def link_layer(self, prev_layer, next_layer):
        super().link_layer(prev_layer, next_layer)

    @abstractmethod
    def _gradient_loss(self, *args, **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass


class InputLayer(_Layer):
    def __init__(self, input_size):
        super().__init__(input_size)

    def prepare_for_training(self, max_batch_size, optimizer, normalizer):
        super().prepare_for_training(max_batch_size, optimizer, normalizer)
        self.o_values = np.zeros(shape=(max_batch_size, self.size))

    def set_data(self, data, input_batch_size):
        self.o_values[:input_batch_size] = data

    def link_layer(self, prev_layer, next_layer):
        return super().link_layer(prev_layer, next_layer)

    def evaluate_layer(self, input_batch_size, inference):
        return super().evaluate_layer(input_batch_size, inference)

    def train_layer(self, input_batch_size):
        return super().train_layer()

    def _get_output(self):
        return self.o_values

    def lock_layer(self):
        return super().lock_layer()


class PerceptronLayer(_ComputeLayer):
    def __init__(self, num_perceptrons, activation_method):
        super().__init__(num_perceptrons)
        self.activation_method = get_activation_function(activation_method)
        self.activation_method_abl = get_activation_function_abl(activation_method)

    def evaluate_layer(self, input_batch_size, inference):
        self.prev_layer.evaluate_layer(input_batch_size, inference)
        self.b_values[:input_batch_size, :] = np.dot(
            self.prev_layer._get_output()[:input_batch_size, :], self.weights.T) + self.bias

        # apply normalization
        self.b_values[:input_batch_size, :] = self.normalizer.normalize(
            self.b_values[:input_batch_size, :], inference)

        self.o_values[:input_batch_size, :] = self.activation_method(
            self.b_values[:input_batch_size, :])

    def _gradient_loss(self, input_batch_size) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.output_error[:input_batch_size] = np.dot(
            self.next_layer.error_signals[:input_batch_size], self.next_layer.weights)

        self.error_signals[:input_batch_size] = np.multiply(self.output_error[:input_batch_size], self.activation_method_abl(
            self.b_values[:input_batch_size]))

        # outer
        self.d_weights = np.dot(
            self.error_signals[:input_batch_size].T, self.prev_layer.o_values[:input_batch_size]) / input_batch_size

        return self.d_weights, self.error_signals, self.output_error

    def train_layer(self, input_batch_size):
        # get gradient and median along the batches
        delta_batch, error_individual, output_error_individual = self._gradient_loss(
            input_batch_size)
        delta_batch_bias = np.mean(error_individual, axis=0)

        # apply optimizer to the gradient and calculate new weights
        theta = self.weight_optimizer(delta_batch)
        theta_bias = self.bias_optimizer(delta_batch_bias)

        # train Normalization Layer
        self.normalizer.train(output_error_individual)

        self.weights = np.add(self.weights, theta)
        self.bias = np.add(self.bias, theta_bias)

        self.prev_layer.train_layer(input_batch_size)

    def _get_output(self):
        return self.o_values


class PredictionLayer(_ComputeLayer):
    def __init__(self, num_perceptrons, classes: list):
        super().__init__(num_perceptrons)
        self.classes = classes
        self.activation_method = get_activation_function("softmax")

    def evaluate_layer(self, input_batch_size, inference):
        self.prev_layer.evaluate_layer(input_batch_size, inference)
        self.b_values[:input_batch_size, :] = np.dot(
            self.prev_layer._get_output()[:input_batch_size, :], self.weights.T) + self.bias
        # apply normalization
        self.b_values[:input_batch_size, :] = self.normalizer.normalize(
            self.b_values[:input_batch_size, :], inference)

        self.o_values[:input_batch_size, :] = self.activation_method(
            self.b_values[:input_batch_size, :])

    def _gradient_loss(self, input_batch_size, y_correct) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # loss function: cross entropy with softmax
        self.output_error[:input_batch_size] = np.subtract(
            self.o_values[:input_batch_size], y_correct)
        # works since derivative of softmax is 1
        self.error_signals[:input_batch_size] = self.output_error[:input_batch_size]
        # outer
        self.d_weights = np.dot(
            self.error_signals[:input_batch_size].T, self.prev_layer.o_values[:input_batch_size]) / input_batch_size

        return self.d_weights, self.error_signals, self.output_error

    def train_layer(self, input_batch_size, correct_solution_idx=None, correct_solution=None):
        if correct_solution is not None:
            y_correct = correct_solution
        elif correct_solution_idx is not None:
            y_correct = np.zeros(shape=(input_batch_size, len(self.classes)))
            for i, idx in enumerate(correct_solution_idx):
                y_correct[i, idx] = 1
        else:
            raise ValueError("At least one of the two has to be given")

        cross_entropy = -np.sum(y_correct * np.log(self.o_values)) / input_batch_size
        # print(f"cross entropy: {cross_entropy}")

        # get gradient and median along the batches
        delta_batch, error_individual, output_error_individual = self._gradient_loss(
            input_batch_size, y_correct)
        delta_batch_bias = np.sum(error_individual, axis=0) / input_batch_size
        # print(
        #    f"gradient weights: {np.linalg.norm(delta_batch)} gradient bias: {np.linalg.norm(delta_batch_bias)}")

        # apply optimizer to the gradient and calculate new weights
        theta = self.weight_optimizer(delta_batch)
        theta_bias = self.bias_optimizer(delta_batch_bias)
        # print("Step weight norm:", np.linalg.norm(theta)
        # print("Step bias norm:", np.linalg.norm(theta_bias))

        # train Normalization Layer
        self.normalizer.train(output_error_individual)

        self.weights = np.add(self.weights, theta)
        self.bias = np.add(self.bias, theta_bias)

        self.prev_layer.train_layer(input_batch_size)

    def _get_output(self):
        a = np.argmax(self.o_values, axis=1)
        return [i for i in a]

    def get_prediction(self):
        a = np.argmax(self.o_values, axis=1)
        return [self.classes[i] for i in a]
