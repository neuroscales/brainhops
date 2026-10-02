"""Format-independent fields of world coordinates and displacements."""

__all__ = [
    "LPSCoordinatesField",
    "RASCoordinatesField",
    "homogeneous_matrix",
    "ras_displacement_chain",
    "split_ras_displacement_chain",
]

# dependencies
import numpy as np
import typing_extensions as tx

# core
from brainhops._core import affines as _affines
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend

# datamodel
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.enums import BoundaryCondition

# io
from brainhops.io.base.parsers import WriterError

from .affines import RASToVoxel, VoxelToRAS


class RASCoordinatesField(_xforms.CoordinatesField):
    """Field of RAS coordinates."""

    _input: _systems.CoordinateSystem = _systems.VoxelCoordinateSystem()
    _output: _systems.CoordinateSystem = _systems.RASmm()


class LPSCoordinatesField(_xforms.CoordinatesField):
    """Field of LPS coordinates."""

    _input: _systems.CoordinateSystem = _systems.VoxelCoordinateSystem()
    _output: _systems.CoordinateSystem = _systems.LPSmm()


# ----------------------------------------------------------------------
#   RAS DISPLACEMENT FIELDS
# ----------------------------------------------------------------------
#
# Several formats store a dense field of displacements in world (RAS)
# millimetres, sampled on a grid whose voxel-to-RAS affine they also
# store: NIfTI `DISPVECT` files, and BIDS X5 `nonlinear` transforms. A
# `DisplacementField` adds its values in the units of its own grid, so
# such a field is read as a chain of three transformations -- RAS to
# voxel, the displacements rotated into voxel units, voxel back to RAS --
# and written back by undoing that rotation. Both directions live here,
# so that the formats agree on them.


def ras_displacement_chain(
    vectors: ArrayProtocol,
    vox2ras: np.ndarray,
    *,
    order: tx.Any = 1,
    bound: tx.Any = BoundaryCondition.nearest,
) -> tx.Tuple[RASToVoxel, _xforms.DisplacementField, VoxelToRAS]:
    """
    The chain that maps RAS to RAS through a field of RAS displacements.

    Parameters
    ----------
    vectors : array, shape `(*shape, ndim)`
        Displacements in RAS millimetres, one vector per voxel.
    vox2ras : array, shape `(ndim + 1, ndim + 1)`
        The voxel-to-RAS affine of the grid the vectors are sampled on.
    order, bound
        Interpolation order and boundary condition of the field.

    Returns
    -------
    ras2voxel, displacement, voxel2ras
        Only the linear part of the world-to-voxel affine acts on a
        displacement, so `displacement` holds the vectors rotated into
        voxel units.
    """
    vox2ras = np.asarray(vox2ras, dtype=np.float64)
    ndim = vox2ras.shape[0] - 1
    compact = vox2ras[:ndim]
    backend = get_array_backend(vectors)
    ras2vox = np.linalg.inv(vox2ras[:ndim, :ndim])
    rotate = backend.asarray(ras2vox, dtype=vectors.dtype)
    field = backend.matmul(rotate, vectors[..., None])[..., 0]
    voxel = _systems.VoxelCoordinateSystem()
    return (
        RASToVoxel(matrix=_affines.inv(compact)),
        _xforms.DisplacementField(
            field=field,
            input=voxel,
            output=voxel,
            order=order,
            bound=bound,
        ),
        VoxelToRAS(matrix=compact),
    )


def split_ras_displacement_chain(
    chain: tx.Sequence[_xforms.Transformation],
    what: str = "A displacement field",
    ndim: tx.Optional[int] = None,
) -> tx.Tuple[np.ndarray, ArrayProtocol]:
    """
    Undo [`ras_displacement_chain`][]: the grid and the RAS vectors.

    Parameters
    ----------
    chain : sequence of three transformations
        RAS to voxel, a displacement field in voxel units, voxel to RAS.
    what : str
        How to name the field in error messages.
    ndim : int, optional
        The number of spatial dimensions the format supports.

    Returns
    -------
    vox2ras : array, shape `(ndim + 1, ndim + 1)`
        The voxel-to-RAS affine of the grid, read from the last slot.
    vectors : array, shape `(*shape, ndim)`
        The displacements, rotated back into RAS millimetres.

    Raises
    ------
    WriterError
        If the chain does not have that shape, holds spline
        coefficients, or its grid is not an affine.
    """
    chain = tuple(chain or ())
    if len(chain) != 3 or not isinstance(chain[1], _xforms.DisplacementField):
        raise WriterError(
            f"{what} is written from a chain of three transformations: "
            f"RAS to voxel, a displacement field, and voxel to RAS."
        )
    displacement = chain[1]
    if displacement.field is None:
        raise WriterError(
            "This field has no displacements, so there is nothing to write."
        )
    if displacement.coeff:
        raise WriterError(
            f"{what} stores sampled displacements, and this field holds "
            f"spline coefficients. Convert it to values first."
        )
    vox2ras = homogeneous_matrix(chain[2], what, ndim)
    ndim = vox2ras.shape[0] - 1
    field = displacement.field
    backend = get_array_backend(field)
    field = backend.asarray(field)
    if field.ndim != ndim + 1 or field.shape[-1] != ndim:
        raise WriterError(
            f"{what} holds one {ndim}-vector per voxel of a {ndim}-D "
            f"grid, not an array of shape {tuple(field.shape)}."
        )
    rotate = backend.asarray(vox2ras[:ndim, :ndim], dtype=field.dtype)
    vectors = backend.matmul(rotate, field[..., None])[..., 0]
    return vox2ras, vectors


_NUMBERS = {1: "one", 2: "two", 3: "three", 4: "four"}


def homogeneous_matrix(
    xform: _xforms.Transformation,
    what: str = "A displacement field",
    ndim: tx.Optional[int] = None,
) -> np.ndarray:
    """
    The homogeneous matrix of the transformation that places a field's grid.

    A field stored in world units is read as a chain whose first and last
    transformations map between the field's voxel grid and the world (see
    [`ras_displacement_chain`][]). A writer needs that voxel-to-world
    mapping back as a plain matrix, to store it in the file's header. It
    may come out of the chain as an `Affine`, or as any transformation
    that converts to one, such as a `Scaling`, a `Translation` or a
    format-specific voxel-to-RAS class.

    This function converts `xform` to an `Affine` and returns its
    homogeneous matrix, checking that the format can store it.

    Parameters
    ----------
    xform : Transformation
        The transformation that maps the field's voxel grid to the world,
        usually the last transformation of the chain.
    what : str
        How to name the field in error messages, e.g.
        `"An X5 displacement field"`.
    ndim : int, optional
        The number of spatial dimensions the format supports. When given,
        the matrix must be `(ndim + 1, ndim + 1)`.

    Returns
    -------
    matrix : array, shape `(D + 1, D + 1)`
        The homogeneous voxel-to-world matrix, as float64, where `D` is
        the number of spatial dimensions.

    Raises
    ------
    WriterError
        If `xform` does not convert to an `Affine`, if its matrix is not
        square (the grid and the world must have as many dimensions), or
        if it does not have `ndim` dimensions.
    """
    try:
        matrix = xform.to(_xforms.Affine).homogeneous_matrix
    except Exception as error:
        raise WriterError(
            f"The grid of {what[:1].lower() + what[1:]} must be an affine, "
            f"not a {type(xform).__name__}."
        ) from error
    matrix = np.asarray(matrix, dtype=np.float64)
    square = matrix.shape[0] == matrix.shape[1]
    if not square or (ndim is not None and matrix.shape[0] != ndim + 1):
        dims = _NUMBERS.get(ndim, f"{ndim}") if ndim else "square"
        raise WriterError(
            f"The grid of {what[:1].lower() + what[1:]} must be "
            f"{dims}-dimensional, and its affine has shape "
            f"{matrix.shape[0] - 1}x{matrix.shape[1] - 1}."
        )
    return matrix
