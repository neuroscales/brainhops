"""How the data of a slide level is decoded from `rawdata`, and encoded back.

The lazy array of a level already presents the pixels as brainhops orders
them, an F-ordered `(x, y, c)` RGB array, so the pair of functions below is
the identity, except that the lazy array is turned into the data that the
reader option `lazy` asks for: the lazy array itself, which reads only the
region that it is indexed with, or a dask array of whole tiles over it.
Neither reads a pixel. The two functions are pure and invert each other, so
that `_image_to_model(_image_to_disk(data))` gives `data` back.
"""

# internals
from brainhops._core._internal_api import ArrayProtocol

# this format
from ._data import _SlideArray


def _image_to_model(raw: ArrayProtocol) -> ArrayProtocol:
    """Return the data of a level from the array that `raw` holds.

    A lazy array of a slide gives the view that its option `lazy` asks for,
    and any other array, such as data that was consolidated into `rawdata`, is
    returned as it is. The lazy array cannot be written to, so the data
    decoded from a slide is read-only.
    """
    if isinstance(raw, _SlideArray):
        return raw.view()
    return raw


def _image_to_disk(data: ArrayProtocol) -> ArrayProtocol:
    """Return the array that `rawdata` holds from the data of a level.

    A level stores its data as it is, so the array is returned unchanged.
    """
    return data
