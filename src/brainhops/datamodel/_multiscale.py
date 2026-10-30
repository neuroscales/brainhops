"""The helpers with which multiscale objects build their scales.

These names are internal: the multiscale classes of the data model, and
the formats that build their scales through
`brainhops.datamodel._internal_api`, use them, while the public
method is
[`Multiscale.with_scales`][brainhops.datamodel.multiscale.Multiscale.with_scales].
`PyramidLayout` describes the arrays of the finest scale.
`pop_pyramid_options` and `build_pyramid` take out and apply the options
of a conversion that build the scales. `geometry_matrix`,
`downsampled_matrix` and `level_matches` place a level from the finest
one.
"""

# dependencies
import numpy as np
import typing_extensions as tx

# core
from brainhops._core.pyramids import level_geometry

# datamodel
from .axes import AxisList

PYRAMID_OPTIONS = ("nscales", "factor", "mode", "align", "min_size", "strict")
"""The options of a conversion that build the scales of a multiscale object.

`nscales=` takes the values of the argument `scales` of
[`Multiscale.with_scales`][brainhops.datamodel.multiscale.Multiscale.with_scales],
and the other options are those of that method.
"""


class PyramidLayout(tx.NamedTuple):
    """What a multiscale object tells about the arrays of its finest scale.

    The method
    [`Multiscale.with_scales`][brainhops.datamodel.multiscale.Multiscale.with_scales]
    resolves the factors that it is given against this layout.
    """

    axes: AxisList
    """The axes of the arrays of the finest scale, in their order.

    They only resolve the names that a factor gives, and are never
    stored.
    """

    shape: tx.Tuple[int, ...]
    """The number of voxels of the finest scale along each axis."""

    sizes: tx.Tuple[float, ...]
    """The size of a voxel of the finest scale along each axis."""

    spatial: tx.Tuple[int, ...]
    """The positions of the axes that are downsampled by default."""

    fixed: tx.Tuple[int, ...] = ()
    """The positions of the axes that are never downsampled.

    This is the axis that holds the components of the vectors of a
    field.
    """

    field: bool = False
    """Whether the scales hold a field of displacements or coordinates."""


def pop_pyramid_options(options: tx.Dict[str, tx.Any]) -> tx.Dict[str, tx.Any]:
    """Remove the options that build the scales from those of a conversion.

    Parameters
    ----------
    options : dict
        The options of a conversion, which this function edits.

    Returns
    -------
    dict
        The options of [`PYRAMID_OPTIONS`][] that were among them, which
        may be none.
    """
    return {key: options.pop(key) for key in PYRAMID_OPTIONS if key in options}


def build_pyramid(
    obj: tx.Any, options: tx.Mapping[str, tx.Any], default: bool = False
) -> tx.Any:
    """Give a multiscale object the scales that conversion options ask for.

    Parameters
    ----------
    obj : Multiscale
        The multiscale object that a conversion built.
    options : mapping
        The options that [`pop_pyramid_options`][] took out.
    default : bool, default=False
        Whether the conversion builds the scales without `nscales=`, as
        a conversion from a single scale to a class whose
        [`BUILDS_SCALES`][brainhops.datamodel.multiscale.Multiscale.BUILDS_SCALES]
        is true does.

    Returns
    -------
    Multiscale
        `obj` with its scales, or `obj` itself when nothing is built.

    Raises
    ------
    TypeError
        If options other than `nscales=` are given to a conversion that
        builds nothing, such as one that keeps the scales of a pyramid.
    """
    options = dict(options)
    if "nscales" in options or default:
        return obj.with_scales(options.pop("nscales", None), **options)
    if options:
        names = ", ".join(f"{key}=" for key in sorted(options))
        raise TypeError(
            f"{names} build the scales of a multiscale object, but this "
            "conversion keeps its scales. Give `nscales=` to build them."
        )
    return obj


def geometry_matrix(transform: tx.Any, ndim: int) -> tx.Optional[np.ndarray]:
    """Return the compact matrix of a geometry that is an affine.

    Parameters
    ----------
    transform : Transformation
        The geometry, such as the transformation of a level.
    ndim : int
        The number of axes that the geometry maps onto as many axes.

    Returns
    -------
    ndarray or None
        The matrix of shape `(ndim, ndim + 1)`, which is the identity
        for an affine without a matrix, or `None` when the geometry is
        not an affine of `ndim` axes.
    """
    if transform.is_identity():
        return np.eye(ndim, ndim + 1)
    affine = transform.to_affine(on_error=None)
    if affine is None:
        return None
    if affine.matrix is None:
        return np.eye(ndim, ndim + 1)
    matrix = np.asarray(affine.matrix, dtype=float)
    if matrix.shape != (ndim, ndim + 1) or not np.isfinite(matrix).all():
        return None
    return matrix


def downsampled_matrix(
    matrix: np.ndarray, factors: tx.Sequence[float], shift: tx.Sequence[float]
) -> np.ndarray:
    """Return the placement of a level, from that of the finest level.

    The voxel `i` of the level is the voxel `factors * i + shift` of the
    finest level, which `matrix` places.

    Parameters
    ----------
    matrix : ndarray
        The compact matrix of the finest level.
    factors : sequence of float
        The factor of each axis.
    shift : sequence of float
        The shift of each axis, in voxels of the finest level, as
        [`level_map`][brainhops._core.pyramids.level_map] returns it.

    Returns
    -------
    ndarray
        The compact matrix of the level.
    """
    linear = matrix[:, :-1]
    columns = linear * np.asarray(factors, dtype=float)
    origin = matrix[:, -1] + linear @ np.asarray(shift, dtype=float)
    return np.concatenate([columns, origin[:, None]], axis=1)


def level_matches(
    finest: np.ndarray,
    level: np.ndarray,
    finest_shape: tx.Sequence[int],
    level_shape: tx.Sequence[int],
    factors: tx.Sequence[int],
    align: str,
) -> bool:
    """Return whether a level is the level of `factors` and `align`.

    The level is such a level when its shape is the one that
    [`level_geometry`][brainhops._core.pyramids.level_geometry] gives,
    and when its matrix is that of the finest level after the actual
    scale and the shift of each axis.

    Parameters
    ----------
    finest : ndarray
        The compact matrix that places the finest level.
    level : ndarray
        The compact matrix that places the level.
    finest_shape : sequence of int
        The shape of the finest level.
    level_shape : sequence of int
        The shape of the level.
    factors : sequence of int
        The factor of each axis from the finest level.
    align : str
        The alignment of the level.

    Returns
    -------
    bool
        Whether the level is such a level.
    """
    if not len(finest_shape) == len(level_shape) == len(factors):
        return False
    shape, scales, shifts = level_geometry(align, finest_shape, factors)
    if tuple(int(n) for n in level_shape) != shape:
        return False
    expected = downsampled_matrix(finest, scales, shifts)
    size = max(float(np.abs(finest[:, :-1]).max()), 1e-300)
    return bool(np.allclose(level, expected, rtol=1e-6, atol=1e-6 * size))
