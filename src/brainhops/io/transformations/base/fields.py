"""Format-independent fields of world coordinates and displacements."""

__all__ = [
    "LPSCoordinatesField",
    "RASCoordinatesField",
    "homogeneous_matrix",
    "ras_displacement_chain",
    "split_ras_displacement_chain",
    "voxel_grid_coordinates",
]

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import KwOnly

# core
from brainhops._core import affines as _affines
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend

# datamodel
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.enums import BoundaryCondition, StoreEnum

# io
from brainhops.io.base.parsers import WriterError

from .affines import RASToVoxel, VoxelToRAS


class RASCoordinatesField(_xforms.CoordinatesField):
    """Field of RAS coordinates."""

    _input: KwOnly[_systems.CoordinateSystem] = (
        _systems.VoxelCoordinateSystem()
    )
    _output: KwOnly[_systems.CoordinateSystem] = _systems.RASmm()


class LPSCoordinatesField(_xforms.CoordinatesField):
    """Field of LPS coordinates."""

    _input: KwOnly[_systems.CoordinateSystem] = (
        _systems.VoxelCoordinateSystem()
    )
    _output: KwOnly[_systems.CoordinateSystem] = _systems.LPSmm()


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
#
# The same chain holds a field of B-spline coefficients of RAS
# displacements (an X5 `bspline` transform), with
# `store="coefficients"`: the
# displacement at a point is a linear combination of the coefficients,
# so rotating the coefficients rotates the displacement they encode.


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
    """
    The chain that maps RAS to RAS through a field of RAS displacements.

    Parameters
    ----------
    vectors : array, shape `(*shape, ndim)`
        Displacements in RAS millimetres, one vector per voxel -- or,
        when `store` is `"coefficients"`, the spline coefficients of those
        displacements, one vector per knot. When `log` is set, they are
        the stationary velocity whose flow is the map, rather than its
        displacement.
    vox2ras : array, shape `(ndim + 1, ndim + 1)`
        The voxel-to-RAS affine of the grid the vectors are sampled on.
    degree, bound
        Spline degree and boundary condition of the field.
    store : {"values", "coefficients"}
        Whether `vectors` are spline coefficients rather than values.
    log : bool
        Whether `vectors` are a stationary velocity, which makes the field
        a [`StationaryVelocityField`][brainhops.datamodel.\
transformations.StationaryVelocityField]. A velocity rotates into voxel
        units as a displacement does, and its flow commutes with the change
        of coordinates, so the chain is the same.
    steps : int, optional
        The number of squaring steps of a velocity. Only a velocity takes
        it.

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
    # `steps` is a field of a velocity only: it is passed when it is set, so
    # that it is refused without `log`.
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
    """
    Undo [`ras_displacement_chain`][]: the grid and the RAS vectors.

    The field is first converted to the encoding the format stores, and
    its `data` is what is written. A field already in that encoding is
    passed through as it is: its stored array is written, unrefitted.

    Parameters
    ----------
    chain : sequence of three transformations
        RAS to voxel, a displacement field in voxel units, voxel to RAS.
    what : str
        How to name the field in error messages.
    ndim : int, optional
        The number of spatial dimensions the format supports.
    store : {"values", "coefficients"}
        Whether the format stores spline coefficients rather than
        sampled values. A field of values written to a format of
        coefficients is encoded, and the reverse is decoded.
    degree, bound : optional
        The spline degree and boundary condition of the coefficients a
        format stores. A field of coefficients under other ones is
        refitted. `None` keeps the field's own.
    log : bool
        Whether the format stores a stationary velocity rather than a
        displacement. A velocity written to a format of displacements is
        integrated; a displacement written to a format of velocities is
        refused, since a field has no logarithm that brainhops computes.

    Returns
    -------
    vox2ras : array, shape `(ndim + 1, ndim + 1)`
        The voxel-to-RAS affine of the grid, read from the last slot.
    vectors : array, shape `(*shape, ndim)`
        The displacements (or their coefficients), rotated back into RAS
        millimetres.

    Raises
    ------
    WriterError
        If the chain does not have that shape, holds no field, or its
        grid is not an affine.
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
    """
    The world coordinate of every voxel of a grid.

    Formats that store a field of world *positions* (an FSL absolute
    warp, a NiftyReg deformation field or control-point grid) are read
    as displacements by subtracting these, and written back by adding
    them. A `CoordinatesField` would hold the same map inside its grid,
    but no boundary condition extends positions beyond the grid the way
    these tools do -- by keeping the edge *displacement* -- while
    `nearest` on a displacement field does.

    Parameters
    ----------
    shape : sequence of int
        The shape of the grid, `(X, Y, Z)`.
    vox2world : array, shape `(ndim + 1, ndim + 1)` or `(ndim, ndim + 1)`
        The voxel-to-world affine of the grid.
    backend : ArrayBackend, optional
        The array backend to build the coordinates with (NumPy by
        default), so that they match the field they are combined with.

    Returns
    -------
    coordinates : array, shape `(*shape, ndim)`
        The coordinate of voxel `(i, j, k)` is
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
