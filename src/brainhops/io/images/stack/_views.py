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
from brainhops._core.dependencies import da
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend
from brainhops.io.images.base import _utils_raster as raster

# this format
from ._data import _StackProxy
from ._raw import PLANES


def _image_to_model(raw: ArrayProtocol, axis: str = "z") -> ArrayProtocol:
    """Return the data of an image from the stacked files.

    A lazy array is first read with the array backend. With NumPy, every
    file is read, and the array is made read-only, because it is a view
    that is never written back, so an edit in place would be lost. With
    Dask, the array is a Dask array with one chunk per slice, and a file
    is read only when a chunk that it fills is computed. Any other array
    is permuted as it is, so that the data of an image whose data were set
    are a view of `rawdata`.
    """
    if isinstance(raw, _StackProxy):
        backend = get_array_backend()
        if da is not None and backend is da:
            # `name=False` gives a random name instead of a hash of the
            # proxy, which Dask computes only to recognize two arrays of
            # the same source (docstring of `dask.array.from_array`, dask
            # 2023.5.0), and which says nothing about the files.
            raw = da.from_array(
                raw, chunks=raw.chunks, name=False, asarray=True
            )
        else:
            raw = backend.asarray(raw)
            if isinstance(raw, np.ndarray):
                raw.flags.writeable = False
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
