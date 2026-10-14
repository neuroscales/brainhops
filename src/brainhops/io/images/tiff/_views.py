"""How the data of a TIFF image is decoded from the array that the file
stores, and encoded back.

The lazy array of a TIFF level already presents the pixels in the order
of brainhops, whatever the order of the axes in the file, because that
order is given by the record. The pair of functions below is therefore
the identity, except that the lazy array is read as tifffile reads it: a
memory map or a dask array stays as it is, and only a level that must be
decoded is read in full. The two functions are pure and invert each
other, so that `_image_to_model(_image_to_disk(data))` gives `data` back.
"""

# dependencies
import numpy as np

# internals
from brainhops._core.typing import ArrayProtocol

# this format
from ._data import _TiffProxy


def _image_to_model(raw: ArrayProtocol) -> ArrayProtocol:
    """Return the data of an image from the array that the file stores.

    A lazy array is read as a view in the order of brainhops. A NumPy
    array that it gives, such as a memory map, is made read-only, because
    the writer writes `raw` and not the data, so an edit in place would
    be lost. Any other array is returned as it is, so that a dask array
    stays lazy and an array that was set as data stays the array of
    `raw`.
    """
    if not isinstance(raw, _TiffProxy):
        return raw
    # The view is new at each read, so its flag does not change the
    # array that tifffile gave.
    data = raw.read()
    if isinstance(data, np.ndarray):
        data.flags.writeable = False
    return data


def _image_to_disk(data: ArrayProtocol) -> ArrayProtocol:
    """Return the array that the file of an image stores from its data.

    An image stores its data as it is, so the array is returned unchanged.
    """
    return data
