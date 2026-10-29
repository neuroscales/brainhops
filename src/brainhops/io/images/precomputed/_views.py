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
from brainhops._core._internal_api import ArrayProtocol

# this format


def _image_to_model(raw: ArrayProtocol) -> ArrayProtocol:
    """Return the data of an image from the array that the volume stores.

    The array is returned as it is, so that the lazy array of a file
    stays lazy and reads nothing. The `data` of the object returns that
    lazy array, which reads its values only when a computation needs
    them.
    """
    return raw


def _image_to_disk(data: ArrayProtocol) -> ArrayProtocol:
    """Return the array that the volume stores from the data of an image.

    The array is the data itself.
    """
    return data
