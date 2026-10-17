"""A lazy array over the array proxy of nibabel.

The NIfTI and MGH formats read their voxels with nibabel, whose
`ArrayProxy` reads a selection when it is indexed and returns a NumPy
array. The `dataobj` of an object read from a file is a lazy array that
reads nothing until it is converted (see
[`ArrayProxy`][brainhops._core.proxy.ArrayProxy]), so the views of these
formats wrap the nibabel proxy in the class below, which brainhops can
index and transpose without reading anything.
"""

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops._core.proxy import ArrayProxy


class NibabelProxy(ArrayProxy):
    """A lazy array that reads its values through a nibabel proxy.

    The class gives the nibabel proxy the lazy indexing of
    [`ArrayProxy`][brainhops._core.proxy.ArrayProxy]. A box that a view
    selects is read by indexing the nibabel proxy, which reads only the
    bytes of that box and applies the scaling of the header.

    Parameters
    ----------
    proxy : nibabel.arrayproxy.ArrayProxy
        The proxy of nibabel.
    """

    def __init__(self, proxy: tx.Any) -> None:
        self.proxy = proxy
        """The proxy of nibabel."""

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

    def _read(self, index: tx.Tuple[tx.Any, ...]) -> np.ndarray:
        return np.asanyarray(self.proxy[index])
