"""How the data of a TIFF image is decoded from the array that the file
stores, and encoded back.

The lazy array of a TIFF level already presents the pixels in the order
of brainhops, whatever the order of the axes in the file, because that
order is given by the record. The pair of functions below is therefore
the identity. The lazy array reads the level as tifffile reads it when
it is converted: a memory map or a dask array stays as it is, and only a
level that must be decoded is read in full. The two functions are pure
and invert each other, so that `_image_to_model(_image_to_disk(data))`
gives `data` back.
"""

# internals
from brainhops._core.typing import ArrayProtocol

# this format
from ._data import _TiffProxy


def _image_to_model(raw: ArrayProtocol) -> ArrayProtocol:
    """Return the data of an image from the array that the file stores.

    The lazy array of a level already presents the pixels in the order of
    brainhops, so it is returned as it is and reads nothing. When the
    `data` of the image reads it, tifffile gives a memory map, a Dask
    array or the decoded level (see [`_TiffProxy`][]). A level read with
    the reader option `lazy=True` is returned as its Dask array, which
    also reads nothing, so that it stays lazy whatever its size.
    """
    if isinstance(raw, _TiffProxy) and raw.lazy:
        return raw.to_dask()
    return raw


def _image_to_disk(data: ArrayProtocol) -> ArrayProtocol:
    """Return the array that the file of an image stores from its data.

    An image stores its data as it is, so the array is returned unchanged.
    """
    return data
