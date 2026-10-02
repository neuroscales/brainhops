"""
Voxel-to-world geometry helpers shared by file formats.

Several formats (NIfTI, MRtrix, ...) store the geometry of an image as a
three-dimensional voxel-to-RAS affine. The helpers here reduce a
transformation to that matrix. They depend only on the datamodel and
NumPy -- not on nibabel -- so that formats that do not need nibabel can
use them.
"""

__all__ = [
    "RAS_FROM_ORIENTATION",
    "embed_affine",
    "ras_conversion",
    "reduce_to_affine",
]

# externals
import numpy as np
import typing_extensions as tx

# internals
from brainhops.datamodel.systems import CoordinateSystem, _axes_or_unknown
from brainhops.datamodel.transformations import (
    Affine,
    ConversionError,
    Sequence,
    Transformation,
)
from brainhops.io.base.parsers import (
    UnrepresentableTransformationError,
    WriterError,
)

RAS_FROM_ORIENTATION = {
    "left-to-right": (0, 1.0),
    "right-to-left": (0, -1.0),
    "posterior-to-anterior": (1, 1.0),
    "anterior-to-posterior": (1, -1.0),
    "inferior-to-superior": (2, 1.0),
    "superior-to-inferior": (2, -1.0),
}
"""
The RAS axis and sign that an anatomical orientation points along.

Each key is the value of an anatomical orientation carried by an axis. The
first element of the pair is the index of the RAS axis the orientation runs
along, and the second is its sign. This drives the conversion of a
voxel-to-world affine into voxel-to-RAS from the axes themselves, rather
than from the world space's name.
"""


def ras_conversion(system: tx.Optional[CoordinateSystem]) -> np.ndarray:
    """
    The `(4, 4)` matrix that maps a world space's coordinates into RAS.

    The matrix is built from the anatomical orientation carried by each
    axis, not from the world space's name. An LPS space becomes a flip of
    the first two axes, an RSA space becomes a permutation, and a space
    already in RAS becomes the identity.

    The conversion is derived only when all three leading axes carry a
    recognized anatomical orientation. When any of them does not, the
    identity is returned, so a space with no orientation is stored as it
    is.
    """
    # An axis about which nothing is known, including the `...` of a
    # missing space, carries no orientation.
    axes = _axes_or_unknown(system)[:3]
    mapping = []
    for axis in axes:
        value = getattr(getattr(axis, "orientation", None), "value", None)
        if value not in RAS_FROM_ORIENTATION:
            return np.eye(4)
        mapping.append(RAS_FROM_ORIENTATION[value])
    if len(mapping) != 3:
        return np.eye(4)
    conversion = np.zeros((4, 4))
    conversion[3, 3] = 1.0
    for column, (row, sign) in enumerate(mapping):
        conversion[row, column] = sign
    return conversion


def reduce_to_affine(
    xform: Transformation, fmt: str, world: str = "world"
) -> Affine:
    """
    Reduce a voxel-to-world transformation to an `Affine`.

    An affine transformation is used directly. A transformation of any
    other kind that reduces to an affine, such as a `Scaling` or a
    `Sequence` of affines, is converted.

    Parameters
    ----------
    xform : Transformation
        The voxel-to-world transformation.
    fmt : str
        The name of the format, used in error messages.
    world : str
        The name the format gives its world space, used in error
        messages.

    Raises
    ------
    UnrepresentableTransformationError
        If the transformation has no affine representation, such as a
        displacement field.
    """
    reduced = xform.compute() if isinstance(xform, Sequence) else xform
    error = None
    affine = reduced
    if not isinstance(affine, Affine):
        try:
            affine = reduced.to(Affine)
        except ConversionError as exc:
            error = exc
    if not isinstance(affine, Affine):
        # A field returns itself from a conversion to `Affine`, and a
        # `Sequence` of a non-affine reduces to one, so the result has to
        # be checked rather than trusted.
        raise UnrepresentableTransformationError(
            f"A {type(xform).__name__} cannot be written as {fmt} "
            f"geometry: {fmt} stores an affine voxel-to-{world} matrix, "
            f"and this transformation has no affine representation."
        ) from error
    return affine


def embed_affine(
    matrix: np.ndarray, fmt: str, world: str = "world"
) -> np.ndarray:
    """
    Embed a homogeneous voxel-to-world matrix in a `(4, 4)` matrix.

    A two-dimensional map yields a `(3, 3)` homogeneous matrix, whose
    rotation and translation are placed in a `(4, 4)` matrix whose extra
    axis is the identity. A three-dimensional map is already `(4, 4)` and
    is returned unchanged.

    Parameters
    ----------
    matrix : ndarray
        The homogeneous voxel-to-world matrix.
    fmt : str
        The name of the format, used in error messages.
    world : str
        The name the format gives its world space, used in error
        messages.

    Raises
    ------
    WriterError
        If the map has more than three spatial dimensions.
    """
    out_dim = matrix.shape[0] - 1
    in_dim = matrix.shape[1] - 1
    if out_dim > 3 or in_dim > 3:
        raise WriterError(
            f"{fmt} stores a three-dimensional voxel-to-{world} affine, so "
            f"a {in_dim}D-to-{out_dim}D transformation cannot be written. "
            f"Reduce the transformation to three spatial dimensions before "
            f"writing it to {fmt}."
        )
    embedded = np.eye(4)
    embedded[:out_dim, :in_dim] = matrix[:out_dim, :in_dim]
    embedded[:out_dim, 3] = matrix[:out_dim, in_dim]
    return embedded
