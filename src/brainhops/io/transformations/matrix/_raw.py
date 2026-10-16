"""The record of a plain matrix file, with the conventions it was read in."""

__all__ = ["MatrixAffineRaw"]

import numpy as np
import typing_extensions as tx
from bagof.magic import NoRepr

from brainhops.datamodel import systems as _systems
from brainhops.io.base._raw import Raw
from brainhops.io.base.parsers import ParserContentError

from . import _conventions

_Matrix = tx.Tuple[tx.Tuple[float, ...], ...]


class MatrixAffineRaw(Raw):
    """Record of a plain matrix file, and of the conventions it was read in.

    A plain matrix file holds a small matrix and nothing else, so it is
    read in one pass, and the record holds the whole file: the matrix as
    it is stored, and, for a text file, the text that it was read from.
    The file records nothing of what the matrix means, so the caller
    states its conventions when the file is read, as described in
    [`MatrixAffine`][brainhops.io.transformations.matrix.MatrixAffine],
    and the record keeps them in the form that decodes the matrix. The
    metadata of every plain matrix format, such as
    [`TxtMatrixMetadata`][brainhops.io.transformations.matrix.TxtMatrixMetadata]
    or
    [`NpzMatrixMetadata`][brainhops.io.transformations.matrix.NpzMatrixMetadata],
    holds a record of this type, and a plain matrix format reads its
    matrix and its coordinate systems from that record. A record is never
    changed in place, so that several objects can share it.

    The record of a file that is read describes the file. The record of a
    transformation that is consolidated holds the matrix of the
    transformation, stored in the conventions and the layout of the record
    that it replaces, and no text.
    """

    matrix: _Matrix = ()
    """The matrix as the file stores it, as a tuple of rows of floats.

    The matrix may be of shape (3, 3), (3, 4) or (4, 4) in 3-D, or (2, 3)
    or (3, 3) in 2-D, or the transpose of one of these for row vectors.
    """

    text: NoRepr[tx.Optional[str]] = None
    """The text of a text file, as it was read.

    It is `None` for a binary container and for a record that was encoded
    from a matrix.
    """

    variable: tx.Optional[str] = None
    """The `.npz` key or the `.mat` variable that holds the matrix.

    It is `None` for a container that holds a single array.
    """

    ndim: int = 3
    """The number of spatial dimensions that the matrix acts on, 2 or 3."""

    vector: str = "column"
    """`"column"`, or `"row"` when the stored matrix is applied as `x @ A`."""

    direction: str = "forward"
    """`"forward"`, or `"inverse"` when the matrix maps output to input."""

    index_base: tx.Tuple[int, int] = (0, 0)
    """The index base of the input and of the output, each 0 or 1."""

    source_to_world: tx.Optional[_Matrix] = None
    """The voxel-to-world matrix of the source image, or `None`.

    When it is given, the voxel input of the stored matrix is placed in
    the world space of the source image.
    """

    target_to_world: tx.Optional[_Matrix] = None
    """The voxel-to-world matrix of the target image, or `None`.

    When it is given, the voxel output of the stored matrix is placed in
    the world space of the target image.
    """

    input: tx.Optional[_systems.CoordinateSystem] = None
    """The input coordinate system of the affine that the record holds."""

    output: tx.Optional[_systems.CoordinateSystem] = None
    """The output coordinate system of the affine that the record holds."""

    @classmethod
    def from_array(
        cls,
        array: tx.Any,
        *,
        text: tx.Optional[str] = None,
        variable: tx.Optional[str] = None,
        **conventions,
    ) -> "MatrixAffineRaw":
        """Build the record of a matrix read from a file, in its conventions.

        Parameters
        ----------
        array : array_like
            The two-dimensional matrix that the file stores.
        text : str, optional
            The text of a text file, which the record keeps.
        variable : str, optional
            The name of the matrix in a container of several arrays.
        **conventions : Any
            The conventions of the matrix: `input`, `output`, `source`,
            `target`, `index_base`, `direction`, `vector` and `ndim`, as
            described in
            [`MatrixAffine`][brainhops.io.transformations.matrix.MatrixAffine].

        Returns
        -------
        MatrixAffineRaw
            The record.

        Raises
        ------
        TypeError
            If an option is not a convention.
        ParserContentError
            If the matrix is not the matrix of an affine, or if the
            conventions contradict each other.
        """
        unknown = sorted(set(conventions) - set(_conventions.CONVENTIONS))
        if unknown:
            raise TypeError(
                f"{cls.__name__}.from_array() got unexpected options "
                f"{', '.join(unknown)}; the conventions of a plain matrix "
                f"are {', '.join(_conventions.CONVENTIONS)}."
            )
        array = np.asarray(array)
        try:
            resolved = _conventions.resolve(array, **conventions)
        except ValueError as e:
            raise ParserContentError(str(e)) from e
        return cls(
            matrix=_as_rows(array),
            text=text,
            variable=variable,
            **resolved,
        )

    def affine(self) -> np.ndarray:
        """Return the homogeneous matrix of the affine that the record holds.

        The conventions of the record are applied to its matrix, in the
        order described in
        [`MatrixAffine`][brainhops.io.transformations.matrix.MatrixAffine].

        Returns
        -------
        np.ndarray
            The `(N + 1, N + 1)` matrix that maps column vectors, with
            0-based voxel indices.
        """
        return _conventions.decode(self)


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _as_rows(array: np.ndarray) -> _Matrix:
    """Return a two-dimensional array as a tuple of rows of floats."""
    return tuple(
        tuple(float(v) for v in row)
        for row in np.asarray(array, dtype=np.float64)
    )
