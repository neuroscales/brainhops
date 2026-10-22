"""A lazy array over the array proxy of nibabel.

The NIfTI and MGH formats read their voxels with nibabel, whose
`ArrayProxy` reads a selection when it is indexed and returns a NumPy
array. The `dataobj` of an object read from a file is a lazy array that
reads nothing until it is converted (see
[`ArrayProxy`][brainhops._core.proxy.ArrayProxy]), so the views of these
formats wrap the nibabel proxy in the class below, which brainhops can
index and transpose without reading anything.
"""

# stdlib
import math

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops._core.dependencies import dk
from brainhops._core.proxy import ArrayProxy

_SPATIAL = 3
"""The number of leading axes of a NIfTI or MGH array that are spatial."""

_CHUNK_BYTES = 128 * 2**20
"""The size of a block when Dask is not installed to give its own."""


class NibabelProxy(ArrayProxy):
    """A lazy array that reads its values through a nibabel proxy.

    The class gives the nibabel proxy the lazy indexing of
    [`ArrayProxy`][brainhops._core.proxy.ArrayProxy]. A box that a view
    selects is read by indexing the nibabel proxy, which reads only the
    bytes of that box and applies the scaling of the header.

    A NIfTI or MGH file stores its voxels contiguously, with the first
    axis varying fastest, so the blocks that the proxy prefers are made
    of whole slices along the last spatial axis. Each block holds as
    many slices as fit in the block size of Dask, which is the option
    `array.chunk-size` of its configuration, and every value of the
    axes that follow the spatial axes, such as the time and the vector
    components.
    """

    proxy: tx.Annotated[tx.Any, tx.Doc("The proxy of nibabel.")]

    @property
    def base_shape(self) -> tx.Tuple[int, ...]:
        """The shape of the stored array."""
        return tuple(int(n) for n in self.proxy.shape)

    @property
    def dtype(self) -> np.dtype:
        """The type of the values once the scaling of the header applies.

        The type is computed as nibabel computes it, by scaling an empty
        array of the stored type, so no voxel is read.
        """
        from nibabel.volumeutils import apply_read_scaling

        empty = np.empty(0, self.proxy.dtype)
        return apply_read_scaling(
            empty, self.proxy.slope, self.proxy.inter
        ).dtype

    @property
    def base_chunks(self) -> tx.Tuple[int, ...]:
        """The block shape: whole slices along the last spatial axis.

        A block holds every value of the other axes, and as many slices
        as fit in the block size of Dask, with at least one slice.
        """
        shape = self.base_shape
        if not shape:
            return shape
        last = min(_SPATIAL, len(shape)) - 1
        slice_bytes = math.prod(shape[:last] + shape[last + 1 :])
        slice_bytes *= np.dtype(self.dtype).itemsize
        slices = max(1, _chunk_bytes() // max(1, slice_bytes))
        return shape[:last] + (min(slices, shape[last]),) + shape[last + 1 :]

    def _read(self, index: tx.Tuple[tx.Any, ...]) -> np.ndarray:
        return np.asanyarray(self.proxy[index])


def _chunk_bytes() -> int:
    """Return the number of bytes of a block that Dask is configured for."""
    if dk is None:
        return _CHUNK_BYTES
    from dask.utils import parse_bytes

    size = dk.config.get("array.chunk-size", _CHUNK_BYTES)
    return int(parse_bytes(size)) if isinstance(size, str) else int(size)
