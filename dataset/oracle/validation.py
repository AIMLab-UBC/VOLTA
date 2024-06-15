import numpy as np
from scipy import ndimage


def _erosion_validation_func(seg):
    if not all(np.all(np.take(seg, index, axis=axis) == 0) for axis in range(seg.ndim) for index in (0, -1)):
        return False
    erosion = ndimage.binary_erosion((seg != 0).astype(int), structure=np.ones((4, 4)))
    if not erosion.any():
        return False
    return True
