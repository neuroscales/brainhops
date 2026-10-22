"""
Spline fields as the formats of registration tools store them.

Several registration tools store a nonlinear transformation as the
coefficients of a B-spline on a grid of control points: FNIRT with
`--cout`, NiftyReg with `reg_f3d -cpp`, and elastix with a
`BSplineTransform`. This module holds the helpers that the converters
into these formats share, in addition to those of
[`_conversions`][brainhops.io.transformations.base._conversions].

A transformation that holds one field between two affines is the map
`T = A o (id + d) o B`, where `B` carries the world to the grid of the
field, `d` is the displacement in the voxels of that grid, and `A`
carries the grid to the output world. When the field is a B-spline of
the degree that a format stores, the format holds this map without loss
([`exact_spline_parts`][]), because a displacement is linear in its
coefficients and the affines can be moved around the field:

- `T = S o (A B)`, where `S` is the same field placed on the grid that
  `A` describes, and `A B` is an affine applied first;
- `T = (A B) o S'`, where `S'` is the same field placed on the grid that
  the inverse of `B` describes, and `A B` is an affine applied last.

Any other transformation is sampled on a grid of control points and its
coefficients are fitted to the samples by least squares
([`fit_spline`][]). The fit is a loss, which the converter reports
with a [`LossyConversionError`][brainhops.errors.LossyConversionError] that
carries the result. When the samples are the nodes of the grid, the fit
is the spline interpolation of the values at the nodes.

How a field is extended beyond its grid, and how the boundary condition
of a spline changes the coefficients that lie just outside the grid, is
not counted as a loss, as in the rest of the conversions: each format
extends its field in its own way.
"""

__all__ = [
    "SplineParts",
    "control_grid",
    "exact_spline_parts",
    "fit_spline",
    "least_squares",
    "orthonormal_placement",
    "source_grid",
]

import numpy as np
import typing_extensions as tx

from brainhops._core.bsplines import spline_matrix
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.systems import CoordinateSystem, VoxelCoordinateSystem
from brainhops.errors import ConversionError

from ._conversions import (
    _bridge,
    _is_field,
    _leaves,
    _matrix,
    apply_affine,
    sample_on_grid,
    split_field_chain,
    undoes,
    unrepresentable,
)

_VOXEL = VoxelCoordinateSystem()


class SplineParts(tx.NamedTuple):
    """
    A transformation taken apart around the coefficients of a B-spline.

    The map is `T(x) = after(before(x) + d(before(x)))`, where `d` is
    the B-spline whose coefficients are `coefficients`, in the voxels of
    its grid. The parts are what [`exact_spline_parts`][] and
    [`fit_spline`][] return.
    """

    before: np.ndarray
    """The homogeneous affine from the input world to the grid of the
    coefficients, of shape `(D + 1, D + 1)`."""

    coefficients: tx.Any
    """The coefficients of the displacement, or of the velocity, in the
    voxels of the grid, as an array of shape `(*shape, D)`."""

    after: np.ndarray
    """The homogeneous affine from the grid to the output world."""

    degree: int
    """The degree of the B-spline."""

    loss: tx.Optional[str] = None
    """What the spline loses against the transformation, as the end of
    a sentence, or `None` when it is the same map inside its grid."""

    steps: tx.Optional[int] = None
    """The number of squaring steps with which a velocity is integrated,
    or `None` for the default rule or for a displacement."""

    @property
    def sandwich(self) -> bool:
        """Whether `after` undoes `before`, so that no affine is left
        around the field."""
        return undoes(self.before, self.after)


def exact_spline_parts(
    t: _xforms.Transformation,
    input: CoordinateSystem,
    output: CoordinateSystem,
    cls: type,
    degrees: tx.Container[int],
    log: bool = False,
) -> tx.Optional[SplineParts]:
    """
    Take `t` apart around a B-spline that a format stores as it is.

    The transformation must hold one field between affines, once its
    endpoints are bridged to `input` and `output`. The field must be a
    field of displacements, or a velocity when `log` is true, whose
    degree is one of `degrees` and whose data is known. The coefficients
    are then read without being fitted again: a field that stores its
    values is converted to coefficients only above degree 1, where the
    two differ.

    Parameters
    ----------
    t : Transformation
        The transformation to take apart.
    input : CoordinateSystem
        The world that the format maps from.
    output : CoordinateSystem
        The world that the format maps to.
    cls : type
        The format, which names the error when a bridge does not exist.
    degrees : container of int
        The degrees of B-spline that the format stores.
    log : bool, default=False
        Whether the format stores a velocity rather than a displacement.

    Returns
    -------
    SplineParts or None
        The parts, or `None` when `t` has no such form.
    """
    try:
        before, field, after = split_field_chain(t, input, output, cls)
    except ConversionError:
        return None
    if not isinstance(field, _xforms.DisplacementField):
        return None
    if field.data is None or bool(field.log) != bool(log):
        return None
    degree = int(field.degree)
    if degree not in degrees:
        return None
    if degree > 1:
        coefficients = field.to(store="coefficients").data
    else:
        coefficients = field.values
    steps = getattr(field, "steps", None) if log else None
    return SplineParts(before, coefficients, after, degree, steps=steps)


def source_grid(
    t: _xforms.Transformation, input: CoordinateSystem, cls: type
) -> tx.Tuple[np.ndarray, tx.Tuple[int, ...]]:
    """
    Return the grid of the first field of `t`, placed in `input`.

    A converter that fits a B-spline to `t` places its control points on
    this grid when the format does not impose one, so that the fit has
    the resolution of the field that it replaces.

    Parameters
    ----------
    t : Transformation
        The transformation, which holds at least one field.
    input : CoordinateSystem
        The world in which the grid is placed.
    cls : type
        The format, which names the error.

    Returns
    -------
    grid2world : array, shape `(D + 1, D + 1)`
        The homogeneous affine from the voxels of the grid to `input`.
    shape : tuple of int
        The shape of the grid.

    Raises
    ------
    ConversionError
        If `t` holds no field with data, or the field is not reached
        from `input` through affines.
    """
    leaves = [_bridge(input, t.input, t, cls), *_leaves(t)]
    for i, leaf in enumerate(leaves):
        if not _is_field(leaf):
            continue
        if leaf.data is None:
            break
        ndim = int(leaf.data.shape[-1])
        before = [
            *leaves[:i],
            _bridge(leaves[i - 1].output, leaf.input, t, cls),
        ]
        world2grid = _matrix(before, t, cls, ndim)
        shape = tuple(int(n) for n in leaf.data.shape[:-1])
        return np.linalg.inv(world2grid), shape
    raise unrepresentable(
        t,
        cls,
        "it holds no field with values, so there is no grid on which "
        "the coefficients of a B-spline could be fitted.",
    )


def control_grid(
    grid2world: np.ndarray,
    shape: tx.Sequence[int],
    spacing: tx.Optional[tx.Any] = None,
) -> tx.Tuple[np.ndarray, tx.Tuple[int, ...]]:
    """
    Return a grid of control points that covers a grid of voxels.

    The grid is laid out as NiftyReg and elastix lay out their grids of
    cubic B-splines: it has the orientation of the grid of voxels, a
    distance of `spacing` millimetres between its control points, and
    one control point before the first voxel. It has
    `ceil(n * dx / s) + 3` control points along an axis of `n` voxels
    of size `dx`, so that the support of the spline covers every voxel.
    With the default spacing, which is the size of the voxels, the
    control points that lie inside the grid are its voxels.

    Parameters
    ----------
    grid2world : array, shape `(D + 1, D + 1)`
        The affine from the voxels of the grid to the world.
    shape : sequence of int
        The shape of the grid of voxels.
    spacing : float or sequence of float, optional
        The distance between the control points, in millimetres. By
        default, it is the size of the voxels.

    Returns
    -------
    grid2world : array, shape `(D + 1, D + 1)`
        The affine from the indices of the control points to the world.
    shape : tuple of int
        The number of control points along each axis.
    """
    grid2world = np.asarray(grid2world, dtype=np.float64)
    ndim = len(shape)
    voxel = np.linalg.norm(grid2world[:ndim, :ndim], axis=0)
    if spacing is None:
        spacing = voxel
    spacing = np.broadcast_to(np.asarray(spacing, dtype=np.float64), (ndim,))
    step = spacing / voxel
    sizes = np.ceil(np.asarray(shape) / step - 1e-9).astype(int) + 3
    scale = np.eye(ndim + 1)
    scale[:ndim, :ndim] = np.diag(step)
    scale[:ndim, ndim] = -step
    return grid2world @ scale, tuple(int(n) for n in sizes)


def orthonormal_placement(
    grid2world: np.ndarray, tolerance: float = 1e-6
) -> tx.Optional[tx.Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Split the affine of a grid into directions, spacing and origin.

    ITK and elastix place a grid by an orthonormal matrix of directions,
    the distance between the nodes along each axis, and the position of
    the first node. An affine whose columns are not orthogonal, such as
    one with a shear, has no such form.

    Parameters
    ----------
    grid2world : array, shape `(D + 1, D + 1)`
        The affine from the voxels of the grid to the world.
    tolerance : float, default=1e-6
        The largest error accepted on the orthonormality of the
        directions.

    Returns
    -------
    tuple of array or None
        The direction `(D, D)`, the spacing `(D,)` and the origin
        `(D,)`, or `None` when the columns are not orthogonal.
    """
    grid2world = np.asarray(grid2world, dtype=np.float64)
    ndim = grid2world.shape[0] - 1
    linear = grid2world[:ndim, :ndim]
    spacing = np.linalg.norm(linear, axis=0)
    if np.any(spacing == 0):
        return None
    direction = linear / spacing
    if not np.allclose(direction.T @ direction, np.eye(ndim), atol=tolerance):
        return None
    return direction, spacing, grid2world[:ndim, ndim].copy()


def fit_spline(
    t: _xforms.Transformation,
    grid2world: np.ndarray,
    shape: tx.Sequence[int],
    input: CoordinateSystem,
    output: CoordinateSystem,
    cls: type,
    degree: int,
    bound: tx.Any,
    refine: tx.Any = 1,
    log: bool = False,
) -> SplineParts:
    """
    Fit a B-spline on a grid of control points to `t`.

    `t` is sampled at the control points and, with `refine` larger than
    one, at `refine - 1` points between two control points along each
    axis. The displacement at each sample, in the units of the grid, is
    the position that the sample maps to, carried back to the grid,
    minus the sample. The coefficients are the least-squares fit of the
    spline to these displacements, which is computed one axis at a time
    because the spline is a product of one-dimensional splines. When
    `refine` is one, there are as many samples as coefficients and the
    fit interpolates the values at the control points.

    A velocity, with `log`, is fitted in the same way, from the values
    of the velocity rather than from the displacement that it
    integrates to. It must then be the only field of `t`, between two
    affines that undo each other.

    Parameters
    ----------
    t : Transformation
        The transformation to fit.
    grid2world : array, shape `(D + 1, D + 1)`
        The affine from the indices of the control points to `input`.
    shape : sequence of int
        The number of control points along each axis.
    input : CoordinateSystem
        The world that the format maps from.
    output : CoordinateSystem
        The world that the format maps to, in which `grid2world` also
        places the grid when the displacement is computed.
    cls : type
        The format, which names the error.
    degree : int
        The degree of the B-spline.
    bound : str or float
        The boundary condition of the spline, with which the
        coefficients beyond the grid are extended.
    refine : int or sequence of int, default=1
        The number of samples per interval between control points.
    log : bool, default=False
        Whether a velocity is fitted.

    Returns
    -------
    SplineParts
        The parts of the fitted spline, whose `before` is the inverse of
        `grid2world`, whose `after` is `grid2world`, and which report a
        loss.

    Raises
    ------
    ConversionError
        If `t` cannot be sampled on the grid, or a velocity is asked of
        a transformation that does not hold one.
    """
    grid2world = np.asarray(grid2world, dtype=np.float64)
    shape = tuple(int(n) for n in shape)
    ndim = len(shape)
    refine = np.broadcast_to(np.asarray(refine, dtype=int), (ndim,))
    refine = np.maximum(refine, 1)
    samples = tuple(int((n - 1) * r + 1) for n, r in zip(shape, refine))
    scale = np.eye(ndim + 1)
    scale[:ndim, :ndim] = np.diag(1.0 / refine)
    sample2world = grid2world @ scale
    target = _velocity_as_displacement(t, input, output, cls) if log else t
    parts = sample_on_grid(
        target, np.linalg.inv(sample2world), samples, input, output, cls
    )
    positions = np.asarray(parts.field.values, dtype=np.float64)
    nodes = [
        np.arange(m, dtype=np.float64) / r for m, r in zip(samples, refine)
    ]
    grid = np.stack(np.meshgrid(*nodes, indexing="ij"), axis=-1)
    displacements = apply_affine(np.linalg.inv(grid2world), positions) - grid
    coefficients = least_squares(displacements, nodes, shape, degree, bound)
    loss = (
        f"its field is fitted with a B-spline of degree {int(degree)} on "
        f"the grid of control points of the format, which only "
        f"approximates it between its samples."
    )
    steps = None
    if log:
        steps = split_field_chain(t, input, output, cls)[1].steps
    return SplineParts(
        np.linalg.inv(grid2world),
        coefficients,
        grid2world,
        int(degree),
        loss,
        steps,
    )


def least_squares(
    values: tx.Any,
    coordinates: tx.Sequence[tx.Any],
    shape: tx.Sequence[int],
    degree: int,
    bound: tx.Any,
) -> np.ndarray:
    """
    Fit the coefficients of a B-spline to values on a separable grid.

    The values are given at the nodes of a grid whose coordinates, in
    the units of the grid of coefficients, are the same along each axis
    for every node, as on a grid that is aligned with the grid of
    coefficients. The B-spline is a product of one-dimensional splines,
    so the least-squares fit is the product of the fits along each axis,
    which are computed one after the other with the pseudo-inverse of
    the matrix of each one-dimensional spline.

    Parameters
    ----------
    values : array, shape `(*samples, D)`
        The values of the field at the samples.
    coordinates : sequence of array
        The coordinates of the samples along each axis, in the units of
        the grid of coefficients.
    shape : sequence of int
        The number of coefficients along each axis.
    degree : int
        The degree of the B-spline.
    bound : str or float
        The boundary condition with which the coefficients are extended
        beyond the grid.

    Returns
    -------
    array, shape `(*shape, D)`
        The coefficients.
    """
    coefficients = np.asarray(values, dtype=np.float64)
    for axis, (n, coords) in enumerate(zip(shape, coordinates)):
        weights = np.asarray(
            spline_matrix(
                int(n),
                np.asarray(coords, np.float64),
                int(degree),
                bound,
                True,
            ),
            dtype=np.float64,
        )
        solve = np.linalg.pinv(weights)
        coefficients = np.moveaxis(
            np.tensordot(solve, coefficients, axes=([1], [axis])), 0, axis
        )
    return coefficients


def _velocity_as_displacement(
    t: _xforms.Transformation,
    input: CoordinateSystem,
    output: CoordinateSystem,
    cls: type,
) -> _xforms.Transformation:
    # A chain in which the velocity of `t` is read as a displacement, so
    # that sampling it gives the values of the velocity rather than of
    # its flow. The flow of a velocity commutes with a change of
    # coordinates only when the affines around it undo each other.
    before, field, after = split_field_chain(t, input, output, cls)
    if not isinstance(field, _xforms.DisplacementField) or not field.log:
        raise unrepresentable(
            t,
            cls,
            "the format stores a velocity, and its field holds a "
            "displacement, whose logarithm brainhops does not compute.",
        )
    if not undoes(before, after):
        raise unrepresentable(
            t,
            cls,
            "its velocity lies between affines that do not undo each "
            "other, so its flow is not the flow of one velocity in the "
            "world.",
        )
    values = _xforms.DisplacementField(
        data=field.data,
        degree=field.degree,
        bound=field.bound,
        store=field.store,
        input=_VOXEL,
        output=_VOXEL,
    )
    return _xforms.Sequence(
        transformations=[
            _xforms.Affine(matrix=before[:-1], input=input, output=_VOXEL),
            values,
            _xforms.Affine(matrix=after[:-1], input=_VOXEL, output=output),
        ]
    )
