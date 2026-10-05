"""
Voxel-to-world geometry helpers shared by file formats.

Several formats (NIfTI, MRtrix, ...) store the geometry of an image as a
three-dimensional voxel-to-RAS affine. The helpers here reduce a
transformation to that matrix, and split off the axes that follow the
spatial ones (such as time), which those formats store apart. They depend
only on the datamodel and NumPy -- not on nibabel -- so that formats that
do not need nibabel can use them.
"""

__all__ = [
    "RAS_FROM_ORIENTATION",
    "embed_affine",
    "ras_conversion",
    "reduce_to_affine",
    "split_spatial",
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


def split_spatial(
    matrix: np.ndarray, fmt: str, world: str = "world", nspace: int = 3
) -> tx.Tuple[np.ndarray, tx.List[tx.Tuple[float, float]]]:
    """
    Split a homogeneous voxel-to-world matrix into its spatial block and
    the scale and offset of each axis that follows the spatial ones.

    A format that stores a spatial affine stores the axes that follow the
    `nspace` spatial ones -- the time axis of a time series -- apart, each
    by a spacing and, for some, an origin. A map over more axes splits
    into those only when it does not couple the spatial axes with the
    others, and maps each other axis onto itself alone: its matrix is
    block-diagonal, with a diagonal second block.

    A map over at most `nspace` axes is returned unchanged, with no other
    axis.

    Parameters
    ----------
    matrix : ndarray
        The homogeneous voxel-to-world matrix, of shape
        `(n_out + 1, n_in + 1)`.
    fmt : str
        The name of the format, used in error messages.
    world : str
        The name the format gives its world space, used in error
        messages.
    nspace : int
        The number of leading axes that are spatial.

    Returns
    -------
    spatial : ndarray
        The homogeneous matrix of the spatial axes, of shape
        `(nspace + 1, nspace + 1)`, or `matrix` unchanged.
    others : list of (float, float)
        The scale and the offset of each axis that follows the spatial
        ones, in order.

    Raises
    ------
    UnrepresentableTransformationError
        If the map couples the spatial axes with the others, mixes two of
        the others, or maps different numbers of axes in and out.
    """
    n_out, n_in = matrix.shape[0] - 1, matrix.shape[1] - 1
    if n_out <= nspace and n_in <= nspace:
        return matrix, []
    if n_out != n_in:
        raise UnrepresentableTransformationError(
            f"{fmt} stores a {nspace}D voxel-to-{world} affine and a spacing "
            f"for each other axis, so a map from {n_in} to {n_out} axes "
            f"cannot be written."
        )
    linear = matrix[:n_out, :n_in]
    others = linear[nspace:, nspace:]
    coupled = (
        np.any(linear[:nspace, nspace:] != 0)
        or np.any(linear[nspace:, :nspace] != 0)
        or np.any(others != np.diag(np.diag(others)))
    )
    if coupled:
        raise UnrepresentableTransformationError(
            f"{fmt} stores a {nspace}D voxel-to-{world} affine over the "
            f"spatial axes and a spacing for each other axis (such as "
            f"time), so a map that mixes the spatial axes with the others, "
            f"or two of the others, cannot be written."
        )
    spatial = np.eye(nspace + 1)
    spatial[:nspace, :nspace] = matrix[:nspace, :nspace]
    spatial[:nspace, nspace] = matrix[:nspace, n_in]
    extra = [
        (float(matrix[d, d]), float(matrix[d, n_in]))
        for d in range(nspace, n_out)
    ]
    return spatial, extra
