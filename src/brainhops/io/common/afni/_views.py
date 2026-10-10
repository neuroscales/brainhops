"""How the data of an AFNI image is decoded from the array that the file
stores, and encoded back.

The lazy array of a BRIK already has the shape of the image: it drops
the axis of sub-bricks when there is a single one, and it applies the
scale factors of the header when its values are read. Both depend on
the record, so they belong to the lazy array and not to the pair of
functions below, which is therefore the identity, except that the lazy
array is read with the array backend. The backend loads it with NumPy
and keeps it lazy with Dask. The two functions are pure and invert each
other, so that `_image_to_model(_image_to_disk(data))` gives `data`
back.
"""

# dependencies
import numpy as np

# internals
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend

# this format
from ._data import _BrikProxy


def _image_to_model(raw: ArrayProtocol) -> ArrayProtocol:
    """Return the data of an image from the array that the file stores.

    A lazy array is read with the array backend. The NumPy array that it
    gives is made read-only, because it is a view that the writer does
    not read back: the writer writes `raw`, so an edit in place would be
    lost. Any other array is returned as it is, so that a Dask array
    stays lazy and an array that was set as data stays the array of
    `raw`.
    """
    if not isinstance(raw, _BrikProxy):
        return raw
    data = get_array_backend().asarray(raw)
    if isinstance(data, np.ndarray):
        data.flags.writeable = False
    return data


def _image_to_disk(data: ArrayProtocol) -> ArrayProtocol:
    """Return the array that the file of an image stores from its data.

    An image stores its data as it is, so the array is returned unchanged.
    """
    return data
