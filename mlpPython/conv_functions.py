"""
All implementation in this file are from https://numbersmithy.com/2d-and-3d-convolutions-using-numpy/
Special thanks to Guangzhi for providing these implementations
"""

import numpy as np

from line_profiler import LineProfiler
lp = LineProfiler()


def padArray(var, pad1, pad2=None) -> np.ndarray:
    '''Pad array with 0s

    Args:
        var (ndarray): 2d or 3d ndarray. Padding is done on the first 2 dimensions.
        pad1 (int): number of columns/rows to pad at left/top edges.
    Keyword Args:
        pad2 (int): number of columns/rows to pad at right/bottom edges.
            If None, same as <pad1>.
    Returns:
        var_pad (ndarray): 2d or 3d ndarray with 0s padded along the first 2
            dimensions.
    '''
    if pad2 is None:
        pad2 = pad1
    if pad1+pad2 == 0:
        return var
    var_pad = np.zeros(tuple(pad1+pad2+np.array(var.shape[:2])) + var.shape[2:])
    var_pad[pad1:-pad2, pad1:-pad2] = var

    return var_pad


def pickStrided(var, stride) -> np.ndarray:
    '''Pick sub-array by stride
    Args:
        var (ndarray): 2d or 3d ndarray.
        stride (int): stride/step along the 1st 2 dimensions to pick
            elements from <var>.
    Returns:
        result (ndarray): 2d or 3d ndarray picked at <stride> from <var>.
    '''
    if stride < 0:
        raise Exception("<stride> should be >=1.")
    if stride == 1:
        result = var
    else:
        result = var[::stride, ::stride, ...]
    return result


@lp
def conv3D(var, kernel, stride=1, pad=0) -> np.ndarray:
    '''3D convolution by sub-matrix summing.

    Args:
        var (ndarray): 2d or 3d array to convolve along the first 2 dimensions.
        kernel (ndarray): 2d or 3d kernel to convolve. If <var> is 3d and <kernel>
            is 2d, create a dummy dimension to be the 3rd dimension in kernel.
    Keyword Args:
        stride (int): stride along the 1st 2 dimensions. Default to 1.
        pad (int): number of columns/rows to pad at edges.
    Returns:
        result (ndarray): convolution result.
    '''
    var_ndim = np.ndim(var)
    ny, nx = var.shape[:2]
    ky, kx = kernel.shape[:2]
    result = 0
    if pad > 0:
        var_pad = padArray(var, pad, pad)
    else:
        var_pad = var

    for ii in range(ky*kx):
        yi, xi = divmod(ii, kx)
        slabii = var_pad[yi:2*pad+ny-ky+yi+1:1,
                         xi:2*pad+nx-kx+xi+1:1, ...]*kernel[yi, xi]
        if var_ndim == 3:
            slabii = slabii.sum(axis=-1)
        result += slabii

    if stride > 1:
        result = pickStrided(result, stride)

    return np.asarray(result)


@lp
def conv3DmultDim(var, kernel) -> np.ndarray:
    """wrapper around conv3D to allow the user to plug in higher dimensional arrays"""

    vs = np.shape(var)
    ks = np.shape(kernel)

    result = np.zeros(shape=(vs[0], vs[1], vs[2], ks[0]))

    # the first dimension is the batch index
    # for i in range(vs[0]):
    #    # the second dimension that is looped is the output dimension
    #    for j in range(ks[0]):
    #        result[i, :, :, j] = conv3D(var[i, :, :, :], kernel[j, :, :, :], pad=1)

    return result
