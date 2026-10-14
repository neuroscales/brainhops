"""How the data of a NRRD image is decoded from the array that the file
stores, and encoded back.

The lazy array of a NRRD image already presents the samples in the axis
order of the image, spatial axes first, whatever the order of the file.
That order depends on the record, so it belongs to the lazy array and
not to the pair of functions below, which is therefore the identity,
except that the lazy array is read with the array backend. The backend
loads it with NumPy and keeps it lazy with Dask. The two functions are
pure and invert each other, so that `_image_to_model(_image_to_disk(data))`
gives `data` back.
"""

# dependencies
import numpy as np

# internals
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend
from brainhops.io.common.nrrd._data import _NrrdProxy


def _image_to_model(raw: ArrayProtocol) -> ArrayProtocol:
    """Return the data of an image from the array that the file stores.

    A lazy array is read with the array backend. The NumPy array that it
    gives is made read-only, because it is a view that the writer does
    not read back: the writer writes `rawdata`, so an edit in place
    would be lost. Any other array is returned as it is, so that a Dask
    array stays lazy and an array that was set as data stays the array
    of `rawdata`.
    """
    if not isinstance(raw, _NrrdProxy):
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
