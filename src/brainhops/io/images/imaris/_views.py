"""How the data of an Imaris image is decoded from the array that the file
stores, and encoded back.

The lazy array of an Imaris level presents its voxels slowest first, as
`(channels, time points, z, y, x)`. The data of an image are in the
brainhops order, `(x, y, z, t, c)`, which is the reverse, so the two
functions below reverse the axes. The reversal does not depend on the
record, so it belongs to this pair of pure functions rather than to the
lazy array. The functions invert each other, so that
`_image_to_model(_image_to_disk(data))` gives `data` back.
"""

# dependencies
import numpy as np

# internals
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend

# this format
from ._data import _ImarisProxy


def _image_to_model(raw: ArrayProtocol) -> ArrayProtocol:
    """Return the data of an image from the array that the file stores.

    The axes are reversed. A lazy array is first read with the array
    backend, which loads it with NumPy and keeps it lazy with Dask. The
    NumPy array that it gives is made read-only, because the file cannot
    be written, so an edit in place could never be saved. Any other array
    is reversed as it is, so that the data of an image whose data were
    set is a view of `raw`.
    """
    if isinstance(raw, _ImarisProxy):
        raw = get_array_backend().asarray(raw)
        if isinstance(raw, np.ndarray):
            raw.flags.writeable = False
    return raw.T


def _image_to_disk(data: ArrayProtocol) -> ArrayProtocol:
    """Return the array that the file of an image stores from its data.

    The axes are reversed, to the order of the lazy array.
    """
    return data.T
