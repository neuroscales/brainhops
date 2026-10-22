"""
The grids on which formats of B-splines place their control points.

Several registration tools store a nonlinear transformation as the
coefficients of a B-spline on a grid of control points: FNIRT with
`--cout`, NiftyReg with `reg_f3d -cpp`, and elastix with a
`BSplineTransform`. Such a grid is a field like any other, interpolated
with a B-spline of the degree of the format, so its converters take a
transformation apart with the helpers of
[`_conversions`][brainhops.io.transformations.base._conversions], as
every other field format does, with the degree of the format: a field
of that degree keeps its coefficients, and a field of another degree is
evaluated at its nodes and converted to coefficients, which is a loss.
This module holds what only these formats need: the grid of the field
that a transformation holds, the grid of control points that NiftyReg
and elastix lay over it, and the orthonormal placement that ITK and
elastix require.
"""

__all__ = [
    "control_grid",
    "orthonormal_placement",
    "source_grid",
]

import numpy as np
import typing_extensions as tx

from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.systems import CoordinateSystem

from ._conversions import (
    _bridge,
    _is_field,
    _leaves,
    _matrix,
    unrepresentable,
)


def source_grid(
    t: _xforms.Transformation, input: CoordinateSystem, cls: type
) -> tx.Tuple[np.ndarray, tx.Tuple[int, ...]]:
    """
    Return the grid of the first field of `t`, placed in `input`.

    A converter into a grid of control points lays its grid over this
    one when it is given a spacing of its own, so that the new grid
    covers the field that it replaces.

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
        "it holds no field with values, so there is no grid over which "
        "a grid of control points could be laid.",
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
