"""How the data of a JPEG 2000 image is decoded from the samples that
Pillow decodes, and encoded back.

Pillow decodes a level as `(rows, columns)`, or `(rows, columns,
samples)` for several samples per pixel. The data of an image is indexed
`(x, y)` or `(x, y, c)`, as for every raster image, so the two functions
below swap the first two axes. The proxy that decodes the samples is
read with the array backend, which decodes it with NumPy and keeps it
lazy with Dask. The two functions are pure and invert each other, so
that `image_to_model(image_to_disk(data))` gives `data` back.
"""

# core
from brainhops._core.backends import array_backend
from brainhops._core.typing import ArrayProtocol


def image_to_model(raw: ArrayProtocol) -> ArrayProtocol:
    """Return the data of an image from the samples that Pillow decodes.

    The data are the samples with the columns first. The axes are swapped
    with a transpose, so a lazy array stays lazy and reads nothing.
    """
    perm = [1, 0] + list(range(2, len(raw.shape)))
    return raw.transpose(perm)


def image_to_disk(data: ArrayProtocol) -> ArrayProtocol:
    """Return the samples, as Pillow decodes them, from the data.

    The rows come first, then the columns, then the samples.
    """
    perm = [1, 0] + list(range(2, len(data.shape)))
    return array_backend(data).transpose(data, perm)
