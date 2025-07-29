import numpy as np
from abc import ABC, abstractmethod
from typing import Any

from .activation_functions import get_activation_function, get_activation_function_abl

rng = np.random.default_rng(seed=1)


class _Layer(ABC):
    def __init__(self, size: int):
        self.size = size
        pass

    @abstractmethod
    def prepare_for_training(self, max_batch_size):
        self.max_batch_size = max_batch_size

    @abstractmethod
    def link_layer(self, prev_layer, next_layer):
        self.prev_layer = prev_layer
        self.next_layer = next_layer

    @abstractmethod
    def evaluate_layer(self, input_batch_size):
        pass

    @abstractmethod
    def train_layer(self, *args, **kwargs):
        pass

    @abstractmethod
    def _get_output(self) -> Any:
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

    def prepare_for_training(self, max_batch_size):
        super().prepare_for_training(max_batch_size)
        self.b_values = np.zeros(shape=(max_batch_size, self.size))
        self.o_values = np.zeros(shape=(max_batch_size, self.size))
        self.error_signals = np.zeros(shape=(max_batch_size, self.size))
        self.d_weights = np.zeros(shape=(max_batch_size, self.size, self.prev_layer.size))
        self.weights = rng.normal(0, np.sqrt(2.0 / self.prev_layer.size),
                                  size=(self.size, self.prev_layer.size))

        self.adam_mt_old = np.zeros(shape=(self.size, self.prev_layer.size))
        self.adam_vt_old = np.zeros(shape=(self.size, self.prev_layer.size))

    def link_layer(self, prev_layer, next_layer):
        super().link_layer(prev_layer, next_layer)

    @abstractmethod
    def _gradient_loss(self, *args, **kwargs) -> np.ndarray:
        pass

    def _adam_optimizer(self, adam_g):
        self.adam_time += 1
        m_t = self.adam_beta1 * self.adam_mt_old + (1-self.adam_beta1) * adam_g
        v_t = self.adam_beta2 * self.adam_vt_old + (1-self.adam_beta2) * np.square(adam_g)
        m_corrected = m_t / (1 - (self.adam_beta1 ** self.adam_time))
        v_corrected = v_t / (1 - (self.adam_beta2 ** self.adam_time))

        theta = self.adam_alpha * (m_corrected / (np.sqrt(v_corrected + self.adam_epsilon)))
        self.adam_mt_old = m_t
        self.adam_vt_old = v_t

        return theta


class InputLayer(_Layer):
    def __init__(self, input_size):
        super().__init__(input_size)

    def prepare_for_training(self, max_batch_size):
        super().prepare_for_training(max_batch_size)
        self.o_values = np.zeros(shape=(max_batch_size, self.size))

    def set_data(self, data, input_batch_size):
        self.o_values[:input_batch_size] = data

    def link_layer(self, prev_layer, next_layer):
        return super().link_layer(prev_layer, next_layer)

    def evaluate_layer(self, input_batch_size):
        return super().evaluate_layer(input_batch_size)

    def train_layer(self, input_batch_size):
        return super().train_layer()

    def _get_output(self):
        return self.o_values


class PerceptronLayer(_ComputeLayer):
    def __init__(self, num_perceptrons, activation_method):
        super().__init__(num_perceptrons)
        self.activation_method = get_activation_function(activation_method)
        self.activation_method_abl = get_activation_function_abl(activation_method)

    def evaluate_layer(self, input_batch_size):
        self.prev_layer.evaluate_layer(input_batch_size)
        # self.b_values[:input_batch_size,
        #              :] = self.prev_layer._get_output()[:input_batch_size, :] @ self.weights.T
        self.b_values[:input_batch_size, :] = np.dot(
            self.prev_layer._get_output()[:input_batch_size, :], self.weights.T)
        # np.einsum("ij, aj -> ai", self.weights, self.prev_layer._get_output()
        #          [:input_batch_size, :], out=self.b_values[:input_batch_size])
        self.o_values[:input_batch_size, :] = self.activation_method(
            self.b_values[:input_batch_size, :])

    def _gradient_loss(self, input_batch_size) -> np.ndarray:
        # dot product
        # np.einsum("aj, ji -> ai", self.next_layer.error_signals[:input_batch_size],
        #          self.next_layer.weights, out=self.error_signals[:input_batch_size])
        self.error_signals[:input_batch_size] = np.dot(
            self.next_layer.error_signals[:input_batch_size], self.next_layer.weights)
        np.multiply(self.error_signals[:input_batch_size], self.activation_method_abl(
            self.b_values[:input_batch_size]), out=self.error_signals[:input_batch_size])

        # outer
        # np.einsum("ai, aj -> aij", self.error_signals[:input_batch_size],
        #          self.prev_layer.o_values[:input_batch_size], out=self.d_weights[:input_batch_size])
        self.d_weights[:input_batch_size] = np.multiply(self.error_signals[:input_batch_size, :, None],
                                                        self.prev_layer.o_values[:input_batch_size, None, :])

        return self.d_weights

    def train_layer(self, input_batch_size):
        # get gradient and median along the batches
        delta_individual = self._gradient_loss(input_batch_size)
        delta_batch = np.sum(delta_individual, axis=0)
        delta_batch = np.divide(delta_batch, input_batch_size)

        # apply adam optimizer to the gradient and calculate new weights
        self.weights = np.subtract(self.weights, self._adam_optimizer(delta_batch))

        self.prev_layer.train_layer(input_batch_size)

    def _get_output(self):
        return self.o_values


class PredictionLayer(_ComputeLayer):
    def __init__(self, num_perceptrons, classes: list):
        super().__init__(num_perceptrons)
        self.classes = classes
        self.activation_method = get_activation_function("softmax")

    def evaluate_layer(self, input_batch_size):
        self.prev_layer.evaluate_layer(input_batch_size)
        self.b_values[:input_batch_size, :] = np.dot(
            self.prev_layer._get_output()[:input_batch_size, :], self.weights.T)
        self.o_values[:input_batch_size, :] = self.activation_method(
            self.b_values[:input_batch_size, :])

    def _gradient_loss(self, input_batch_size, y_correct) -> np.ndarray:
        # loss function: cross entropy with softmax
        self.error_signals[:input_batch_size] = np.subtract(
            self.o_values[:input_batch_size], y_correct)
        # outer
        self.d_weights[:input_batch_size] = np.multiply(self.error_signals[:input_batch_size, :, None],
                                                        self.prev_layer.o_values[:input_batch_size, None, :])

        return self.d_weights

    def train_layer(self, input_batch_size, correct_solution_idx=None, correct_solution=None):
        if correct_solution is not None:
            y_correct = correct_solution
        elif correct_solution_idx is not None:
            y_correct = np.zeros(shape=(input_batch_size, len(self.classes)))
            for i, idx in enumerate(correct_solution_idx):
                y_correct[i, idx] = 1
        else:
            raise ValueError("At least one of the two has to be given")

        error = np.sum(0.5 * np.square(y_correct - self.o_values))
        cross_entropy = -np.sum(y_correct * np.log(np.clip(self.o_values, 1e-12, 1.-1e-12)))
        print(f"error: {error}")
        print(f"cross entropy: {cross_entropy}")

        # get gradient and median along the batches
        delta_individual = self._gradient_loss(input_batch_size, y_correct)
        delta_batch = np.sum(delta_individual, axis=0)
        delta_batch = np.divide(delta_batch, input_batch_size)

        # apply adam optimizer to the gradient and calculate new weights
        self.weights = np.subtract(self.weights, self._adam_optimizer(delta_batch))

        self.prev_layer.train_layer(input_batch_size)

    def _get_output(self):
        a = np.argmax(self.o_values, axis=1)
        return [self.classes[i] for i in a]
