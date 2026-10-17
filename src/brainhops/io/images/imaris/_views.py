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

# internals
from brainhops._core.typing import ArrayProtocol

# this format


def _image_to_model(raw: ArrayProtocol) -> ArrayProtocol:
    """Return the data of an image from the array that the file stores.

    The axes are reversed with a transpose, so that the lazy array
    of a file stays lazy and reads nothing. The `data` of the object
    reads it once, as the backend setting says.
    """
    return raw.T


def _image_to_disk(data: ArrayProtocol) -> ArrayProtocol:
    """Return the array that the file of an image stores from its data.

    The axes are reversed, to the order of the lazy array.
    """
    return data.T
