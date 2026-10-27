"""How the data of an image stack is decoded from the stacked files, and
encoded back.

The lazy array of a stack stacks its files along a first axis, as
`(slices, rows, columns[, channels])`. The data of the image are in the
brainhops order, `(x, y, z[, c])`, so the two functions below permute the
three spatial axes and keep the channel axis last, as `to_canonical` and
`to_storage` of the raster conventions do. Which axis of the volume the
first axis of the files is, and which axes their rows and columns are,
depends on the stacking axis of the record (`StackRaw.axes`), which the
functions take as an argument. With the default stacking axis, z, the
stored volume is `(z, y, x[, c])`. The functions are pure and invert each
other, so that `_image_to_model(_image_to_disk(data, axis), axis)` gives
`data` back.
"""

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops._core.typing import ArrayProtocol
from brainhops.io.images.base import _utils_raster as raster

# this format
from ._raw import PLANES


def _image_to_model(raw: ArrayProtocol, axis: str = "z") -> ArrayProtocol:
    """Return the data of an image from the stacked files.

    The axes are permuted with a transpose, so that the lazy array of
    the files stays lazy and reads nothing. The `data` of the object
    returns that lazy array, which reads its values only when a
    computation needs them.
    """
    return raw.transpose(_permutation(raw.ndim, axis))


def _image_to_disk(data: ArrayProtocol, axis: str = "z") -> ArrayProtocol:
    """Return the stacked files of an image from its data.

    The spatial axes are permuted into the order of the stored volume, and
    the channel axis stays last.
    """
    return data.transpose(np.argsort(_permutation(data.ndim, axis)).tolist())


def _permutation(ndim: int, axis: str = "z") -> tx.List[int]:
    """Return the permutation from the stacked files to the data.

    Element `i` is the axis of the stored volume that becomes axis `i` of
    the data. An array of three axes is a volume, and an array of four
    axes has channels last. An array of fewer than three axes is reversed.
    """
    if ndim < 3:
        return list(reversed(range(ndim)))
    codes = axis.upper() + PLANES[axis] + "S" * (ndim - 3)
    return raster.canonical_permutation(codes)
