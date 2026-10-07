
import numpy as np
import math

from abc import ABC, abstractmethod


class _Pool(ABC):
    def __init__(self, original_shape, kernel_size) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.original_shape = original_shape

    @abstractmethod
    def _pool(self, mat, inference) -> np.ndarray:
        pass

    @abstractmethod
    def gradient(self, prev_grad) -> np.ndarray:
        pass

    def __call__(self, *args, **kwds):
        return self._pool(*args, **kwds)


class MaxPool(_Pool):
    def __init__(self, original_shape, kernel_size) -> None:
        super().__init__(original_shape, kernel_size)

        self.grad_mask: np.ndarray

    # from https://stackoverflow.com/questions/42463172/how-to-perform-max-mean-pooling-on-a-2d-array-using-numpy from Jason
    def _pooling_grad(self, mat, ksize, pad=False, inference=True):
        '''Non-overlapping pooling on 2D or 3D data.

        <mat>: ndarray, input array to pool.
        <ksize>: tuple of 2, kernel size in (ky, kx).
        <method>: str, 'max for max-pooling,
                       'mean' for mean-pooling.
        <pad>: bool, pad <mat> or not. If no pad, output has size
               n//f, n being <mat> size, f being kernel size.
               if pad, output has size ceil(n/f).

        Return <result>: pooled matrix and precomputed values for gradient step
        '''

        batch_size, m, n = mat.shape[:3]
        ky, kx = ksize

        def _ceil(x, y): return int(np.ceil(x/float(y)))

        if pad:
            ny = _ceil(m, ky)
            nx = _ceil(n, kx)
            size = (ny*ky, nx*kx)+mat.shape[2:]
            mat_pad = np.full(size, np.nan)
            mat_pad[:, :m, :n, ...] = mat
        else:
            # trim mat to full divisions
            ny = math.floor(m/ky)
            nx = math.floor(n/kx)
            mat_pad = mat[:, :ny*ky, :nx*kx, ...]

        new_shape = (batch_size, ny, nx, ky, kx)+mat.shape[3:]
        reshaped = mat_pad.reshape(new_shape)

        if inference:
            result = np.nanmax(reshaped, axis=(3, 4))
            return result, self.grad_mask
        else:
            max_vals = np.nanmax(reshaped, axis=(3, 4), keepdims=True)
            result = max_vals.squeeze(axis=(3, 4))

            mask = (reshaped == max_vals).astype(int)
            return result, mask

    def _pool(self, mat, inference) -> np.ndarray:
        pooled, self.grad_mask = self._pooling_grad(mat, self.kernel_size, inference=inference)

        return pooled

    def gradient(self, prev_grad) -> np.ndarray:
        result = np.zeros((len(prev_grad),) + self.original_shape)

        batch, iy, ix, cy, cx, cc = np.where(self.grad_mask == 1)
        iy2 = iy*self.kernel_size[0]
        ix2 = ix*self.kernel_size[1]
        iy2 = iy2+cy
        ix2 = ix2+cx
        values = prev_grad[batch, iy, ix, cc].flatten()
        result[batch, iy2, ix2, cc] = values

        return result


class MeanPool(_Pool):
    def __init__(self, original_shape, kernel_size) -> None:
        super().__init__(original_shape, kernel_size)

    def _pool(self, mat, inference) -> np.ndarray:
        return super()._pool(mat, inference)

    def gradient(self, prev_grad) -> np.ndarray:
        return super().gradient(prev_grad)
