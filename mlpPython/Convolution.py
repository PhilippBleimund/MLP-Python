from .Layer import _Layer
from .conv_functions import conv3D, conv3DmultDim

from .Pooling import MaxPool, MeanPool
from .Optimizer import AdamOptimizer, SGDOptimizer

from abc import ABC, abstractmethod

import numpy as np
import math

rng = np.random.default_rng(seed=1)


class _Conv(ABC):
    def __init__(self, size:  tuple) -> None:
        self.size = size

        # for linter. The start and end of an model are None
        self.prev_conv: _Conv
        self.next_conv: _Conv

    @abstractmethod
    def prepare_for_training(self, *args, **kwargs):
        pass

    def link_layer(self, prev_layer, next_layer):
        self.prev_conv = prev_layer
        self.next_conv = next_layer

    @abstractmethod
    def evaluate_layer(self, inference: bool) -> np.ndarray:
        pass

    @abstractmethod
    def train_layer(self, *args, **kwargs):
        pass


class FlatteningLayer(_Conv, _Layer):
    """
    convert multidimensional tensors of convolution layers to singe vector
    """

    def __init__(self) -> None:
        pass

    def prepare_for_training(self, optimizer):
        # this
        self.size = self.prev_conv.size[0] * self.prev_conv.size[1] * self.prev_conv.size[2]
        # or
        # self.size = self.next_layer.size
        pass

    def link_layer(self, prev_layer, next_layer):
        self.prev_conv = prev_layer
        self.next_layer = next_layer

    def evaluate_layer(self, inference: bool) -> np.ndarray:
        prev_conv_tensor = self.prev_conv.evaluate_layer(inference)

        self.o_values = np.reshape(prev_conv_tensor, (len(prev_conv_tensor), self.next_layer.size))

        return self.o_values

    def train_layer(self, propagated_error):
        # nothing to train
        # no own error
        # unflatten error for cnn
        self.error = np.reshape(propagated_error, (len(propagated_error),) + self.prev_conv.size)

        self.prev_conv.train_layer(self.error)


class ConvolutionLayer(_Conv):
    def __init__(self, size: tuple) -> None:
        super().__init__(size)
        self.kernel: np.ndarray

    def prepare_for_training(self, optimizer):
        self.kernel = rng.normal(0, np.sqrt(2 / (self.prev_conv.size[2] * self.size[2] * 6)),
                                 size=(self.size[2], self.prev_conv.size[2], 3, 3))

        if optimizer == "adam":
            self.optimizer = AdamOptimizer(shape=(self.size + self.prev_conv.size))
        elif optimizer == "sgd":
            self.optimizer = SGDOptimizer(shape=(self.size + self.prev_conv.size))

    def evaluate_layer(self, inference: bool) -> np.ndarray:
        prev_conv_tensor = self.prev_conv.evaluate_layer(inference)

        conv_tensor = conv3DmultDim(prev_conv_tensor, self.kernel)
        return conv_tensor

    def _gradient_loss(self, prev_error_signal):
        # flip kernel around both axis
        flipped_kernel = np.flip(self.kernel, axis=(2, 3))
        # swap in/out channels
        flipped_kernel = np.transpose(flipped_kernel, (1, 0, 2, 3))

        self.error = conv3DmultDim(prev_error_signal, flipped_kernel)
        return self.error

    def train_layer(self, propagated_error):
        # apply own error to signal
        next_propagated_error = self._gradient_loss(propagated_error)

        delta_kernel = self.optimizer(next_propagated_error)

        self.kernel = np.add(self.kernel, delta_kernel)

        self.prev_conv.train_layer(next_propagated_error)


class Pool(_Conv):
    def __init__(self, stride: tuple = (3, 3), size: tuple = (), method="max") -> None:
        super().__init__(size)
        self.stride = stride
        self.method = method

    def prepare_for_training(self, optimizer):
        self.size = (int(self.prev_conv.size[0] / self.stride[0]),
                     int(self.prev_conv.size[1] / self.stride[1]), self.prev_conv.size[2])

        if self.method == "max":
            self.pooler = MaxPool(self.prev_conv.size, self.stride)
        elif self.method == "mean":
            self.pooler = MeanPool(self.prev_conv.size, self.stride)

    def evaluate_layer(self, inference: bool) -> np.ndarray:
        prev_conv_tensor = self.prev_conv.evaluate_layer(inference)

        self.o_values = self.pooler(prev_conv_tensor, inference)

        return self.o_values

    def _gradient_loss(self, prev_error_signal):
        self.error = self.pooler.gradient(prev_error_signal)

        return self.error

    def train_layer(self, propagated_error):
        # nothing to train

        # apply own error
        propagated_error = self._gradient_loss(propagated_error)

        self.prev_conv.train_layer(propagated_error)
