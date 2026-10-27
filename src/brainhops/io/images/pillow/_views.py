"""How the data of a raster image is decoded from the array that the file
stores, and encoded back.

Pillow decodes a frame as `(rows, columns)`, or `(rows, columns,
samples)`, and the lazy array of a raster image presents the pixels in
that order. The data of the image are `(x, y)`, or `(x, y, c)`, so that
`data[x, y]` is the pixel of column `x` and row `y`, and the two
functions below swap the first two axes. The swap does not depend on the
record, so it belongs to this pair of pure functions rather than to the
lazy array. The functions invert each other, so that
`_image_to_model(_image_to_disk(data))` gives `data` back.
"""

# internals
from brainhops._core.typing import ArrayProtocol

# this format


def _image_to_model(raw: ArrayProtocol) -> ArrayProtocol:
    """Return the data of an image from the array that the file stores.

    The rows and the columns are swapped with a transpose, so that the
    lazy array of a file stays lazy and reads nothing. The `data` of the
    object returns that lazy array, which reads its values only when a
    computation needs them.
    """
    return _swapped(raw)


def _image_to_disk(data: ArrayProtocol) -> ArrayProtocol:
    """Return the array that the file of an image stores from its data.

    The first two axes are swapped, to the rows and the columns of the
    file.
    """
    return _swapped(data)


def _swapped(array: ArrayProtocol) -> ArrayProtocol:
    """Swap the first two axes of an array, and keep any other axis."""
    ndim = len(array.shape)
    if ndim < 2:
        return array
    return array.transpose((1, 0) + tuple(range(2, ndim)))
