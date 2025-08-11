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
