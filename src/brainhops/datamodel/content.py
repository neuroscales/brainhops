"""What the entries along an axis hold, such as the components of vectors.

Most axes of an image enumerate independent values, such as the colours
of an RGB image, and their `content` is `None`. Some axes hold instead
the components of a geometric object, such as a displacement, the
gradient of an image or a diffusion tensor. When the image is moved or
resampled, these components must be re-expressed with the Jacobian of
the transformation, and the content of the axis says how:

| Class               | `type`               | Action of a linear map `J` |
| ------------------- | -------------------- | -------------------------- |
| [`Vector`][]        | `"vector"`           | `v ↦ J v`                  |
| [`Covector`][]      | `"covector"`         | `g ↦ J⁻ᵀ g`                |
| [`SymmetricMatrix`][] | `"symmetric-matrix"` | `T ↦ J T Jᵀ`             |

The components are expressed in a coordinate system, called the frame
of the values, which the image records in
[`SingleScaleImage.frames`][brainhops.datamodel.images.SingleScaleImage.frames].
The entries of a vector axis follow axes of the system of that frame, in
order (see
[`SingleScaleImage.reframe`][brainhops.datamodel.images.SingleScaleImage.reframe]
for how these axes are found). A full matrix is stored as two vector
axes, whose actions combine into `J T Jᵀ`.

The design is recorded in `spec/design/vector-and-tensor-images.md`.
"""

__all__ = ["Content", "Vector", "Covector", "SymmetricMatrix"]

import numpy as np
import typing_extensions as tx
from bagof.magic import HideIfDefault, NoRepr

from brainhops._core.typing import ArrayProtocol
from brainhops.backends import _is_dask, get_array_backend

from .base import DataModelBase

_Pairs: tx.TypeAlias = tx.Tuple[tx.Tuple[int, int], ...]


class Content(DataModelBase, polymorphic=True, eq=True, repr=HideIfDefault):
    """What the entries along an axis hold.

    Calling `Content` with a `type` builds the subclass of that type, so
    that a mapping such as `{"type": "vector"}` describes a [`Vector`][].
    A plain `Content` describes entries whose nature is unknown, and it
    cannot be re-expressed in another frame.

    Parameters
    ----------
    type : str, optional
        The name of the content, such as `"vector"`.
    """

    type: tx.Optional[str] = None
    """The name of the content, such as `"vector"`."""

    def frame_ndim(self, size: int) -> int:
        """Return the number of axes of the frame that the entries follow.

        Parameters
        ----------
        size : int
            The number of entries along the axis.

        Returns
        -------
        int
            The number of axes of the frame that the entries follow.

        Raises
        ------
        NotImplementedError
            If the content does not say how its entries relate to a frame.
        """
        raise NotImplementedError(
            f"A {type(self).__name__} does not say how its entries relate "
            f"to a coordinate system."
        )

    def apply(
        self, values: ArrayProtocol, matrix: ArrayProtocol, axis: int
    ) -> ArrayProtocol:
        """Apply a linear map of the frame to the entries along an axis.

        Parameters
        ----------
        values : ArrayProtocol
            The values of the image.
        matrix : ArrayProtocol
            The linear map, of shape `(*batch, n, n)`, where `batch`
            broadcasts against the shape of `values` without `axis`.
        axis : int
            The axis of `values` whose entries are transformed.

        Returns
        -------
        ArrayProtocol
            The transformed values, with the shape of `values`.

        Raises
        ------
        NotImplementedError
            If the content does not say how its entries transform.
        """
        raise NotImplementedError(
            f"A {type(self).__name__} does not say how its entries change "
            f"when the coordinate system changes."
        )


class Vector(Content, on={"type": "vector"}):
    """Entries that are the components of a vector.

    A displacement, a velocity or a fibre direction is a vector. When the
    frame changes by a linear map `J`, the components become `J v`.

    Parameters
    ----------
    type : {"vector"}, default="vector"
        Pinned content type.
    """

    type: NoRepr[...]

    def frame_ndim(self, size: int) -> int:
        """Return the number of axes of the frame, one per entry.

        Parameters
        ----------
        size : int
            The number of entries along the axis.

        Returns
        -------
        int
            The number of axes of the frame that the entries follow.
        """
        return size

    def apply(
        self, values: ArrayProtocol, matrix: ArrayProtocol, axis: int
    ) -> ArrayProtocol:
        """Return the components `J v`.

        Parameters
        ----------
        values : ArrayProtocol
            The values of the image.
        matrix : ArrayProtocol
            The linear map `J`, of shape `(*batch, n, n)`.
        axis : int
            The axis of `values` that holds the components.

        Returns
        -------
        ArrayProtocol
            The transformed values.
        """
        return _matvec(values, matrix, axis)


class Covector(Content, on={"type": "covector"}):
    """Entries that are the components of a covector.

    The gradient of an image and the normal of a surface are covectors.
    When the frame changes by a linear map `J`, the components become
    `J⁻ᵀ g`, so that the pairing of a covector with a vector does not
    change.

    Parameters
    ----------
    type : {"covector"}, default="covector"
        Pinned content type.
    """

    type: NoRepr[...]

    def frame_ndim(self, size: int) -> int:
        """Return the number of axes of the frame, one per entry.

        Parameters
        ----------
        size : int
            The number of entries along the axis.

        Returns
        -------
        int
            The number of axes of the frame that the entries follow.
        """
        return size

    def apply(
        self, values: ArrayProtocol, matrix: ArrayProtocol, axis: int
    ) -> ArrayProtocol:
        """Return the components `J⁻ᵀ g`.

        Parameters
        ----------
        values : ArrayProtocol
            The values of the image.
        matrix : ArrayProtocol
            The linear map `J`, of shape `(*batch, n, n)`.
        axis : int
            The axis of `values` that holds the components.

        Returns
        -------
        ArrayProtocol
            The transformed values.
        """
        inverse = _batched(_inv, matrix)
        return _matvec(values, _swap_last(inverse), axis)


class SymmetricMatrix(Content, on={"type": "symmetric-matrix"}):
    """Entries that are the packed components of a symmetric matrix.

    A diffusion tensor is a symmetric matrix. Only the upper triangle is
    stored, and `order` gives the row and the column of each entry, since
    software packs the triangle in different orders. When the frame
    changes by a linear map `J`, the matrix becomes `J T Jᵀ`.

    | Software | `order` (for a frame of three axes)                    |
    | -------- | ------------------------------------------------------ |
    | FSL      | `((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))`     |
    | MRtrix   | `((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))`     |

    Parameters
    ----------
    order : tuple of (int, int), optional
        The row and the column of each entry, with the row not greater
        than the column. By default, the triangle is stored row by row,
        as FSL does.

    Other Parameters
    ----------------
    type : {"symmetric-matrix"}, default="symmetric-matrix"
        Pinned content type.
    """

    type: NoRepr[...]

    order: tx.Optional[_Pairs] = None
    """The row and the column of each entry, row by row by default."""

    def pairs(self, size: int) -> _Pairs:
        """Return the row and the column of each of `size` entries.

        Parameters
        ----------
        size : int
            The number of entries along the axis.

        Returns
        -------
        tuple of (int, int)
            The row and the column of each entry.

        Raises
        ------
        ValueError
            If `size` entries cannot pack a symmetric matrix, or if
            `order` does not list each entry of the upper triangle once.
        """
        ndim = self.frame_ndim(size)
        if self.order is None:
            return tuple((i, j) for i in range(ndim) for j in range(i, ndim))
        pairs = tuple((int(i), int(j)) for i, j in self.order)
        expected = {(i, j) for i in range(ndim) for j in range(i, ndim)}
        if len(pairs) != size or set(pairs) != expected:
            raise ValueError(
                f"The order {pairs} does not list each entry of the upper "
                f"triangle of a {ndim}x{ndim} matrix once."
            )
        return pairs

    def frame_ndim(self, size: int) -> int:
        """Return the size `n` of a matrix with `size` packed entries.

        Parameters
        ----------
        size : int
            The number of entries along the axis, `n (n + 1) / 2`.

        Returns
        -------
        int
            The number of axes of the frame that the entries follow.

        Raises
        ------
        ValueError
            If `size` is not the number of entries of a triangle.
        """
        ndim = int(round((np.sqrt(8 * size + 1) - 1) / 2))
        if ndim * (ndim + 1) // 2 != size:
            raise ValueError(
                f"{size} entries cannot hold the upper triangle of a "
                f"square matrix."
            )
        return ndim

    def apply(
        self, values: ArrayProtocol, matrix: ArrayProtocol, axis: int
    ) -> ArrayProtocol:
        """Return the packed components of `J T Jᵀ`.

        Parameters
        ----------
        values : ArrayProtocol
            The values of the image.
        matrix : ArrayProtocol
            The linear map `J`, of shape `(*batch, n, n)`.
        axis : int
            The axis of `values` that holds the packed components.

        Returns
        -------
        ArrayProtocol
            The transformed values.
        """
        xp = get_array_backend(values)
        pairs = self.pairs(values.shape[axis])
        ndim = self.frame_ndim(len(pairs))
        packed = xp.moveaxis(values, axis, -1)
        # Unpack into a full symmetric matrix, transform it and pack the
        # upper triangle again.
        index = np.zeros((ndim, ndim), dtype=int)
        for k, (i, j) in enumerate(pairs):
            index[i, j] = index[j, i] = k
        full = packed[..., index.reshape(-1)]
        full = full.reshape(packed.shape[:-1] + (ndim, ndim))
        matrix = _like(matrix, values)
        full = xp.matmul(xp.matmul(matrix, full), _swap_last(matrix))
        rows = [i for i, _ in pairs]
        cols = [j for _, j in pairs]
        flat = full.reshape(full.shape[:-2] + (ndim * ndim,))
        packed = flat[..., [i * ndim + j for i, j in zip(rows, cols)]]
        return xp.moveaxis(packed, -1, axis)


# --- helpers ----------------------------------------------------------


def _like(matrix: ArrayProtocol, values: ArrayProtocol) -> ArrayProtocol:
    """Return `matrix` in the backend of `values`, in a floating type."""
    xp = get_array_backend(values)
    dtype = np.result_type(values.dtype, np.float32)
    if isinstance(matrix, np.ndarray) and not _is_dask(values):
        return xp.asarray(matrix.astype(dtype))
    return matrix.astype(dtype)


def _matvec(
    values: ArrayProtocol, matrix: ArrayProtocol, axis: int
) -> ArrayProtocol:
    """Multiply each vector along `axis` by `matrix`."""
    xp = get_array_backend(values)
    vectors = xp.moveaxis(values, axis, -1)
    matrix = _like(matrix, values)
    out = xp.matmul(matrix, vectors[..., None])[..., 0]
    return xp.moveaxis(out, -1, axis)


def _swap_last(matrix: ArrayProtocol) -> ArrayProtocol:
    """Transpose the last two axes of a stack of matrices."""
    xp = get_array_backend(matrix)
    return xp.swapaxes(matrix, -1, -2)


def _batched(
    function: tx.Callable[[ArrayProtocol], ArrayProtocol],
    matrix: ArrayProtocol,
) -> ArrayProtocol:
    """Apply a function of stacks of matrices to an array of any backend.

    The function receives a NumPy or CuPy array. Dask has no linear
    algebra on stacks of matrices, so the function is applied to each
    block, with the last two axes in a single block.
    """
    if _is_dask(matrix):
        whole = matrix.rechunk({matrix.ndim - 2: -1, matrix.ndim - 1: -1})
        return whole.map_blocks(function, dtype=matrix.dtype)
    return function(matrix)


def _inv(matrix: ArrayProtocol) -> ArrayProtocol:
    """Invert a stack of matrices."""
    return get_array_backend(matrix).linalg.inv(matrix)
