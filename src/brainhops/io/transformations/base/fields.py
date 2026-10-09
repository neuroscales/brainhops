"""Format-independent fields of world coordinates and displacements."""

__all__ = [
    "LPSCoordinatesField",
    "RASCoordinatesField",
    "homogeneous_matrix",
    "ras_displacement_chain",
    "split_ras_displacement_chain",
    "voxel_grid_coordinates",
]

import numpy as np
import typing_extensions as tx
from bagof.magic import KwOnly

from brainhops._core import affines as _affines
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.enums import BoundaryCondition, StoreEnum
from brainhops.io.base.parsers import WriterError

from .affines import RASToVoxel, VoxelToRAS


class RASCoordinatesField(_xforms.CoordinatesField):
    """Coordinates field from voxel space to RAS millimetres."""

    _input: KwOnly[_systems.CoordinateSystem] = (
        _systems.VoxelCoordinateSystem()
    )
    _output: KwOnly[_systems.CoordinateSystem] = _systems.RASmm()


class LPSCoordinatesField(_xforms.CoordinatesField):
    """Coordinates field from voxel space to LPS millimetres."""

    _input: KwOnly[_systems.CoordinateSystem] = (
        _systems.VoxelCoordinateSystem()
    )
    _output: KwOnly[_systems.CoordinateSystem] = _systems.LPSmm()


# ----------------------------------------------------------------------
#   RAS DISPLACEMENT FIELDS
# ----------------------------------------------------------------------
# Several formats (NIfTI DISPVECT, BIDS X5 nonlinear) store dense RAS
# displacements on a grid placed by a stored vox2ras, whereas a
# `DisplacementField` works in the units of its own grid. Such a field is
# read as RAS to voxel, displacements rotated into voxel units, and voxel
# to RAS, and written by undoing the rotation; keeping both directions here
# makes all formats agree. The chain also serves B-spline coefficients of
# RAS displacements (X5 `bspline`), since a displacement is linear in its
# coefficients.


def ras_displacement_chain(
    vectors: ArrayProtocol,
    vox2ras: np.ndarray,
    *,
    degree: tx.Any = 1,
    bound: tx.Any = BoundaryCondition.nearest,
    store: tx.Any = StoreEnum.values,
    log: bool = False,
    steps: tx.Optional[int] = None,
) -> tx.Tuple[RASToVoxel, _xforms.DisplacementField, VoxelToRAS]:
    """Build a RAS-to-RAS chain through a field of RAS displacements.

    Only the linear part of the world-to-voxel affine acts on a displacement,
    so the field of the chain holds the vectors rotated into voxel units. A
    stationary velocity rotates in the same way and its flow commutes with the
    change of coordinates, so with `log` the same chain holds a
    [`StationaryVelocityField`][brainhops.datamodel.transformations.StationaryVelocityField].

    Parameters
    ----------
    vectors : ArrayProtocol
        Displacements in RAS millimetres, of shape `(*shape, ndim)`, one per
        voxel. With `store="coefficients"` they are spline coefficients, one
        per knot, and with `log` a stationary velocity whose flow is the map.
    vox2ras : np.ndarray
        Square affine of size `ndim + 1` that places the sampling grid.
    degree, bound
        Spline degree and boundary condition of the field.
    store : {"values", "coefficients"}
        Whether `vectors` are values or spline coefficients.
    log : bool
        Whether `vectors` are a stationary velocity.
    steps : int, optional
        Number of squaring steps, which only a velocity field accepts.

    Returns
    -------
    ras2voxel : RASToVoxel
        The world-to-voxel affine.
    displacement : DisplacementField
        The field of vectors rotated into voxel units.
    voxel2ras : VoxelToRAS
        The voxel-to-world affine.
    """
    vox2ras = np.asarray(vox2ras, dtype=np.float64)
    ndim = vox2ras.shape[0] - 1
    compact = vox2ras[:ndim]
    backend = get_array_backend(vectors)
    ras2vox = np.linalg.inv(vox2ras[:ndim, :ndim])
    rotate = backend.asarray(ras2vox, dtype=vectors.dtype)
    field = backend.matmul(rotate, vectors[..., None])[..., 0]
    voxel = _systems.VoxelCoordinateSystem()
    # `steps` is passed only when set, so that a field without `log`
    # refuses it.
    velocity = {} if steps is None else {"steps": steps}
    return (
        RASToVoxel(matrix=_affines.inv(compact)),
        _xforms.DisplacementField(
            data=field,
            input=voxel,
            output=voxel,
            degree=degree,
            bound=bound,
            store=store,
            log=log,
            **velocity,
        ),
        VoxelToRAS(matrix=compact),
    )


def split_ras_displacement_chain(
    chain: tx.Sequence[_xforms.Transformation],
    what: str = "A displacement field",
    ndim: tx.Optional[int] = None,
    store: tx.Any = StoreEnum.values,
    degree: tx.Any = None,
    bound: tx.Any = None,
    log: bool = False,
) -> tx.Tuple[np.ndarray, ArrayProtocol]:
    """Split a RAS displacement chain into its grid and its RAS vectors.

    This function is the inverse of [`ras_displacement_chain`][]. The field
    is first converted to the encoding that the format stores; a field
    already in that encoding passes through, and its stored array is written
    without being refitted.

    Parameters
    ----------
    chain : sequence of Transformation
        RAS to voxel, a displacement field in voxel units, and voxel to RAS.
    what : str
        Name of the written object, used in error messages.
    ndim : int, optional
        Number of spatial dimensions that the format supports.
    store : {"values", "coefficients"}
        Whether the format stores spline coefficients. Values written to a
        coefficient format are encoded, and coefficients written to a value
        format are decoded.
    degree, bound : optional
        Spline degree and boundary condition of the stored coefficients,
        used only with `store="coefficients"`. A coefficient field with other
        settings is refitted, and `None` keeps those of the field.
    log : bool
        Whether the format stores a stationary velocity. A velocity written
        to a displacement format is integrated, while a displacement written
        to a velocity format is refused, since brainhops computes no field
        logarithm.

    Returns
    -------
    vox2ras : np.ndarray
        The `(ndim + 1, ndim + 1)` grid affine, from the last link.
    vectors : ArrayProtocol
        The `(*shape, ndim)` displacements or coefficients, rotated back to
        RAS millimetres.

    Raises
    ------
    WriterError
        If the chain does not have three links around a displacement field,
        holds no displacements, has a grid that is not an affine, or holds an
        array of the wrong shape.
    """
    chain = tuple(chain or ())
    if len(chain) != 3 or not isinstance(chain[1], _xforms.DisplacementField):
        raise WriterError(
            f"{what} is written from a chain of three transformations: "
            f"RAS to voxel, a displacement field, and voxel to RAS."
        )
    encoding: tx.Dict[str, tx.Any] = {"store": store, "log": log}
    coefficients = StoreEnum(store) is StoreEnum.coefficients
    if coefficients and degree is not None:
        encoding["degree"] = degree
    if coefficients and bound is not None:
        encoding["bound"] = bound
    displacement = chain[1].to(**encoding)
    if displacement.data is None:
        raise WriterError(
            "This field has no displacements, so there is nothing to write."
        )
    vox2ras = homogeneous_matrix(chain[2], what, ndim)
    ndim = vox2ras.shape[0] - 1
    field = displacement.data
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


def voxel_grid_coordinates(
    shape: tx.Sequence[int],
    vox2world: np.ndarray,
    backend: tx.Any = None,
) -> ArrayProtocol:
    """Return the world coordinates of every voxel of a grid.

    Formats that store world positions, such as FSL absolute warps and
    NiftyReg deformation or control-point grids, are read as displacements
    by subtracting these coordinates and written by adding them. A
    [`CoordinatesField`][brainhops.datamodel.transformations.CoordinatesField]
    is not used because no boundary condition extends positions beyond the
    grid as these tools do, by keeping the edge displacement, which is what
    `nearest` does on a displacement field.

    Parameters
    ----------
    shape : sequence of int
        Shape of the grid.
    vox2world : np.ndarray
        Affine of shape `(ndim + 1, ndim + 1)` or `(ndim, ndim + 1)`.
    backend : ArrayBackend, optional
        Backend of the result, NumPy by default, to match the field with
        which the coordinates are combined.

    Returns
    -------
    ArrayProtocol
        Coordinates of shape `(*shape, ndim)`; voxel `(i, j, k)` maps to
        `vox2world @ [i, j, k, 1]`.
    """
    if backend is None:
        backend = get_array_backend()
    vox2world = np.asarray(vox2world)
    ndim = len(shape)
    grid = backend.stack(
        backend.meshgrid(*[backend.arange(s) for s in shape], indexing="ij"),
        -1,
    )
    grid = backend.asarray(grid, dtype=vox2world.dtype)
    rotation = backend.asarray(vox2world[:ndim, :ndim], dtype=vox2world.dtype)
    offset = backend.asarray(vox2world[:ndim, ndim], dtype=vox2world.dtype)
    return backend.matmul(rotation, grid[..., None])[..., 0] + offset


_NUMBERS = {1: "one", 2: "two", 3: "three", 4: "four"}


def homogeneous_matrix(
    xform: _xforms.Transformation,
    what: str = "A displacement field",
    ndim: tx.Optional[int] = None,
) -> np.ndarray:
    """Return the homogeneous matrix of the affine that places a field's grid.

    The transformation maps voxels to world, normally as the last link of a
    [`ras_displacement_chain`][]. It may be an
    [`Affine`][brainhops.datamodel.transformations.Affine] or anything that
    converts to one, such as a scaling, a translation or a format-specific
    voxel-to-RAS class. The matrix is checked against what the format can
    store.

    Parameters
    ----------
    xform : Transformation
        The voxel-to-world transformation of the grid.
    what : str
        Name of the written object, used in error messages, for example
        `"An X5 displacement field"`.
    ndim : int, optional
        If given, the number of spatial dimensions that the matrix must have.

    Returns
    -------
    np.ndarray
        A float64 matrix of shape `(D + 1, D + 1)`, with `D` spatial
        dimensions.

    Raises
    ------
    WriterError
        If the transformation does not convert to an affine, if the matrix is
        not square (the grid and the world must have as many dimensions), or
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
