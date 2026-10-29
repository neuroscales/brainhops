"""How the data of a BigDataViewer image is decoded from the array that
the HDF5 file stores, and encoded back.

The HDF5 file stores the voxels of a view in the order z, y, x, and the
lazy array of an image presents them in that order. The data of the
image are in the order x, y, z of ImgLib2, so the two functions below
reverse the axes. The reversal does not depend on the record, so it
belongs to this pair of pure functions rather than to the lazy array.
The functions invert each other, so that
`image_to_model(image_to_disk(data))` gives `data` back.
"""

# internals
from brainhops._core._internal_api import ArrayProtocol


def image_to_model(raw: ArrayProtocol) -> ArrayProtocol:
    """Return the data of an image from the array that the file stores.

    The axes are reversed with a transpose, so that the lazy array of a
    file stays lazy and reads nothing. The `data` of the object returns
    that lazy array, which reads its values only when a computation
    needs them.
    """
    return raw.T


def image_to_disk(data: ArrayProtocol) -> ArrayProtocol:
    """Return the array that the file of an image stores from its data.

    The axes are reversed, to the order z, y, x of the file.
    """
    return data.T
