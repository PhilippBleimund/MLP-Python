from os import wait
from line_profiler import LineProfiler
from .Layer import _Layer
from .conv_functions import conv3D, conv3DmultDim

from abc import ABC, abstractmethod

import numpy as np
import math

rng = np.random.default_rng(seed=1)

lp = LineProfiler()


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
        self.prev_conv.train_layer(propagated_error)

