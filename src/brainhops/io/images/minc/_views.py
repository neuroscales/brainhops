"""How the data of a MINC image is decoded from the array that the file
stores, and encoded back.

A MINC file lists its dimensions slowest first, and the lazy array of a
MINC image presents the voxels in that order. The data of the image are
in Fortran order, fastest axis first, so the two functions below reverse
the axes. The reversal does not depend on the record, so it belongs to
this pair of pure functions rather than to the lazy array. The functions
invert each other, so that `_image_to_model(_image_to_disk(data))` gives
`data` back.
"""

# dependencies
import numpy as np

# internals
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend
from brainhops.io.common.minc._data import _MincProxy


def _image_to_model(raw: ArrayProtocol) -> ArrayProtocol:
    """Return the data of an image from the array that the file stores.

    The axes are reversed. A lazy array is first read with the array
    backend, which loads it with NumPy and keeps it lazy with Dask. The
    NumPy array that it gives is made read-only, because it is a view
    that the writer does not read back, so an edit in place would be
    lost. Any other array is reversed as it is, so that the data of an
    image whose data were set is a view of `raw`.
    """
    if isinstance(raw, _MincProxy):
        raw = get_array_backend().asarray(raw)
        if isinstance(raw, np.ndarray):
            raw.flags.writeable = False
    return raw.T


def _image_to_disk(data: ArrayProtocol) -> ArrayProtocol:
    """Return the array that the file of an image stores from its data.

    The axes are reversed, to the order of the file.
    """
    return data.T
