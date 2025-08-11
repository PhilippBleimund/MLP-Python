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


class ConvolutionLayer(_Conv):
    def __init__(self, size: tuple) -> None:
        super().__init__(size)
        self.kernel: np.ndarray

    def prepare_for_training(self):
        self.kernel = rng.normal(0, np.sqrt(2 / (self.prev_conv.size[2] * self.size[2] * 6)),
                                 size=(self.size[2], self.prev_conv.size[2], 3, 3))

    def evaluate_layer(self, inference: bool) -> np.ndarray:
        prev_conv_tensor = self.prev_conv.evaluate_layer(inference)

        conv_tensor = conv3DmultDim(prev_conv_tensor, self.kernel)
        return conv_tensor

    def train_layer(self, propagated_error):
        self.prev_conv.train_layer(propagated_error)


class MaxPool(_Conv):
    def __init__(self, stride: tuple, size: tuple = ()) -> None:
        super().__init__(size)
        self.stride = stride

    def prepare_for_training(self):
        self.size = (self.prev_conv.size[0] / self.stride[0],
                     self.prev_conv.size[1] / self.stride[1], self.prev_conv.size[2])

    # from https://stackoverflow.com/questions/42463172/how-to-perform-max-mean-pooling-on-a-2d-array-using-numpy from Jason
    def _pooling(self, mat, ksize, method='max', pad=False):
        '''Non-overlapping pooling on 2D or 3D data.

        <mat>: ndarray, input array to pool.
        <ksize>: tuple of 2, kernel size in (ky, kx).
        <method>: str, 'max for max-pooling, 
                       'mean' for mean-pooling.
        <pad>: bool, pad <mat> or not. If no pad, output has size
               n//f, n being <mat> size, f being kernel size.
               if pad, output has size ceil(n/f).

        Return <result>: pooled matrix.
        '''

        m, n = mat.shape[:2]
        ky, kx = ksize

        def _ceil(x, y): return int(np.ceil(x/float(y)))

        if pad:
            ny = _ceil(m, ky)
            nx = _ceil(n, kx)
            size = (ny*ky, nx*kx)+mat.shape[2:]
            mat_pad = np.full(size, np.nan)
            mat_pad[:m, :n, ...] = mat
        else:
            ny = m//ky
            nx = n//kx
            mat_pad = mat[:ny*ky, :nx*kx, ...]

        new_shape = (ny, ky, nx, kx)+mat.shape[2:]

        if method == 'max':
            result = np.nanmax(mat_pad.reshape(new_shape), axis=(1, 3))
        else:
            result = np.nanmean(mat_pad.reshape(new_shape), axis=(1, 3))

        return result

    def evaluate_layer(self, inference: bool) -> np.ndarray:
        prev_conv_tensor = self.prev_conv.evaluate_layer(inference)

        self.o_values = self._pooling(prev_conv_tensor, self.stride)

        return self.o_values
