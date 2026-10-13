"""How the data of an image stack is decoded from the stacked files, and
encoded back.

The lazy array of a stack stacks its files along a first axis, as
`(files, rows, columns[, channels])`, which tifffile spells `ZYX` or
`ZYXS`. The data of the image are in the brainhops order, `(x, y, z[,
c])`, so the two functions below reverse the three spatial axes and
keep the channel axis last, as `to_canonical` of the raster conventions
does for these axes. The permutation does not depend on the record, so it
belongs to this pair of pure functions rather than to the lazy array. It
is its own inverse, so that `_image_to_model(_image_to_disk(data))` gives
`data` back.
"""

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops._core.dependencies import da
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend

# this format
from ._data import _StackProxy


def _image_to_model(raw: ArrayProtocol) -> ArrayProtocol:
    """Return the data of an image from the stacked files.

    A lazy array is first read with the array backend. With NumPy, every
    file is read, and the array is made read-only, because it is a view
    that is never written back, so an edit in place would be lost. With
    Dask, the array is a Dask array with one chunk per file, and a file is
    read only when a chunk that it fills is computed. Any other array is
    permuted as it is, so that the data of an image whose data were set
    are a view of `raw`.
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
    return raw.transpose(_permutation(raw.ndim))


def _image_to_disk(data: ArrayProtocol) -> ArrayProtocol:
    """Return the stacked files of an image from its data.

    The three spatial axes are reversed, and any other axis stays where it
    is.
    """
    return data.transpose(_permutation(data.ndim))


def _permutation(ndim: int) -> tx.List[int]:
    """Return the permutation between the stacked files and the data.

    The first three axes are reversed and the others are kept. An array of
    fewer than three axes is reversed.
    """
    if ndim < 3:
        return list(reversed(range(ndim)))
    return [2, 1, 0] + list(range(3, ndim))
