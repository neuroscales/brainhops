"""How the data of a precomputed scale is decoded from the array that the
volume stores, and encoded back.

The lazy array of a scale already presents the voxels in the brainhops
order, `[x, y, z]` with a channel axis last when the volume has several
channels, because the chunks are Fortran-ordered `[x, y, z, channel]`
arrays. Whether the channel axis is kept depends on the record, so the
lazy array drops it, and the two functions below are the identity. They
invert each other, so that `_image_to_model(_image_to_disk(data))` gives
`data` back.
"""

# internals
from brainhops._core.typing import ArrayProtocol
from brainhops.io.base._utils_proxies import load_proxy

# this format
from ._data import _PrecomputedProxy


def _image_to_model(raw: ArrayProtocol) -> ArrayProtocol:
    """Return the data of an image from the array that the volume stores.

    A lazy array is read with the array backend, which loads it with
    NumPy and keeps it lazy with Dask. The NumPy array that it gives is
    made read-only, because precomputed volumes are not written and an
    edit in place would be lost. Any other array is returned as it is.
    """
    if isinstance(raw, _PrecomputedProxy):
        raw = load_proxy(raw)
    return raw


def _image_to_disk(data: ArrayProtocol) -> ArrayProtocol:
    """Return the array that the volume stores from the data of an image.

    The array is the data itself.
    """
    return data
