"""How the data of an N5 image is decoded from the array that the dataset
stores, and encoded back.

The lazy array of a dataset presents the samples in the order of the
`dimensions` of the dataset, whose first dimension varies fastest in a
block. That order is already the brainhops order, `[x, y, z, ...]`, so
the two functions below are the identity, except that the lazy array is
read with the array backend. They invert each other, so that
`_image_to_model(_image_to_disk(data))` gives `data` back.
"""

# dependencies
import numpy as np

# internals
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend

# this format
from ._data import _N5Proxy


def _image_to_model(raw: ArrayProtocol) -> ArrayProtocol:
    """Return the data of an image from the array that the dataset stores.

    A lazy array is read with the array backend, which loads it with
    NumPy and keeps it lazy with Dask. The NumPy array that it gives is
    made read-only, because N5 datasets are not written and an edit in
    place would be lost. Any other array is returned as it is.
    """
    if isinstance(raw, _N5Proxy):
        raw = get_array_backend().asarray(raw)
        if isinstance(raw, np.ndarray):
            raw.flags.writeable = False
    return raw


def _image_to_disk(data: ArrayProtocol) -> ArrayProtocol:
    """Return the array that the dataset stores from the data of an image.

    The array is the data itself.
    """
    return data
