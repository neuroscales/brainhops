"""The helpers with which multiscale objects build their scales.

These names are internal: the multiscale classes of the data model, and
the formats that build their scales through
`brainhops.datamodel._internal_api`, use them, while the public
method is
[`Multiscale.with_scales`][brainhops.datamodel.multiscale.Multiscale.with_scales].
`PyramidLayout` describes the arrays of the finest scale.
`pop_pyramid_options` and `build_pyramid` take out and apply the options
of a conversion that build the scales. `ScalePlan`, `plan_scales` and
`next_scale` let a writer build the scales one at a time while it
writes them. `geometry_matrix`,
`downsampled_matrix` and `level_matches` place a level from the finest
one.
"""

# stdlib
import functools
from collections.abc import Iterable, Mapping
from numbers import Integral

# dependencies
import numpy as np
import typing_extensions as tx

# core
from brainhops._core.pyramids import (
    DEFAULT_ALIGN,
    DEFAULT_MODE,
    MIN_SIZE,
    check_align,
    check_mode,
    default_factors,
    level_geometry,
)
from brainhops._core.typing import ArrayProtocol

# datamodel
from .axes import AxisList

if tx.TYPE_CHECKING:
    from .multiscale import AxisReference, FactorLike

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


class ScalePlan(tx.NamedTuple):
    """The coarser scales that a pyramid is asked to have.

    [`plan_scales`][] returns a plan from the options of
    [`Multiscale.with_scales`][brainhops.datamodel.multiscale.Multiscale.with_scales],
    and [`next_scale`][] builds the scales of a plan one at a time.
    """

    factors: tx.List[tx.Tuple[int, ...]]
    """The factor of each axis of each coarser scale from the finest."""

    mode: str
    """How a coarse voxel is computed, one of the modes of `with_scales`."""

    align: str
    """How the voxels of a coarse scale lie over those of the finest."""


@functools.singledispatch
def pyramid_layout(obj: tx.Any) -> PyramidLayout:
    """Return the layout of the arrays of the finest scale of a pyramid.

    Each multiscale class that builds its scales registers its version
    of this function.

    Parameters
    ----------
    obj : Multiscale
        The pyramid.

    Returns
    -------
    PyramidLayout
        The layout of its finest scale.

    Raises
    ------
    TypeError
        If the class of `obj` cannot build its scales.
    """
    raise TypeError(
        f"A {type(obj).__name__} cannot build its scales from its finest "
        "scale."
    )


@functools.singledispatch
def build_scales(
    obj: tx.Any,
    factors: tx.Sequence[tx.Tuple[int, ...]],
    mode: str,
    align: str,
    kept: int,
) -> tx.Any:
    """Return a copy of a pyramid that keeps some scales and builds others.

    [`Multiscale.with_scales`][brainhops.datamodel.multiscale.Multiscale.with_scales]
    and [`next_scale`][] call this function, and each multiscale class
    that builds its scales registers its version of it.

    Parameters
    ----------
    obj : Multiscale
        The pyramid.
    factors : sequence of tuple of int
        One tuple of factors, one per axis of the layout, for each
        coarser scale.
    mode : str
        How a coarse voxel is computed.
    align : str
        How the voxels of a coarse scale lie over those of the finest.
    kept : int
        The index of the last scale of `obj` that is kept. The others
        are built from it or from the finest scale.

    Returns
    -------
    Multiscale
        The copy.

    Raises
    ------
    TypeError
        If the class of `obj` cannot build its scales.
    """
    raise TypeError(
        f"A {type(obj).__name__} cannot build its scales from its finest "
        "scale."
    )


@functools.singledispatch
def level_resolution(obj: tx.Any, index: int) -> tx.Optional[ArrayProtocol]:
    """Return the physical grid size of a scale of a pyramid.

    `nearest_level` compares these sizes to choose the scale whose
    resolution is closest to a target grid. Each multiscale class that
    knows the resolution of its scales registers its version of this
    function, and the size of any other scale is unknown.

    Parameters
    ----------
    obj : Multiscale
        The pyramid.
    index : int
        The index of the scale.

    Returns
    -------
    ArrayProtocol or None
        One entry per axis, in the units of the input, or `None` when
        the size is unknown.
    """
    return None


@functools.singledispatch
def keeps_scale(
    obj: tx.Any, index: int, factors: tx.Tuple[int, ...], align: str
) -> bool:
    """Return whether a scale of a pyramid is the one that is asked for.

    [`Multiscale.with_scales`][brainhops.datamodel.multiscale.Multiscale.with_scales]
    keeps a coarser scale that the pyramid already holds when this
    function returns true for it. Each multiscale class that builds its
    scales registers its version of it, and any other scale is built
    again.

    Parameters
    ----------
    obj : Multiscale
        The pyramid.
    index : int
        The index of the scale.
    factors : tuple of int
        The factor of each axis of the layout from the finest scale.
    align : str
        How the voxels of the scale lie over those of the finest scale.

    Returns
    -------
    bool
        Whether the scale lies where `align` puts a scale of `factors`,
        and has its shape.
    """
    return False


@functools.singledispatch
def check_scales(obj: tx.Any, scales: tx.Sequence[tx.Any]) -> None:
    """Refuse a list of scales that a pyramid cannot take as it is.

    [`Multiscale.with_scales`][brainhops.datamodel.multiscale.Multiscale.with_scales]
    calls this function before [`adopt_scales`][]. Each multiscale class
    registers its version of it, and any list passes by default.

    Parameters
    ----------
    obj : Multiscale
        The pyramid.
    scales : sequence
        The scales, finest first.

    Raises
    ------
    TypeError
        If a scale has the wrong type.
    ValueError
        If the scales are not ordered from fine to coarse, or do not
        share one world space.
    """
    return None


@functools.singledispatch
def adopt_scales(obj: tx.Any, scales: tx.Sequence[tx.Any]) -> tx.Any:
    """Return a copy of a pyramid that holds the scales it is given.

    Each multiscale class that can be given a list of scales registers
    its version of this function.

    Parameters
    ----------
    obj : Multiscale
        The pyramid.
    scales : sequence
        The scales, finest first, which [`check_scales`][] accepted.

    Returns
    -------
    Multiscale
        The copy.

    Raises
    ------
    TypeError
        If the class of `obj` cannot be given a list of scales.
    """
    raise TypeError(
        f"A {type(obj).__name__} cannot be given a list of scales."
    )


def plan_scales(obj: tx.Any, options: tx.Mapping[str, tx.Any]) -> ScalePlan:
    """Return the coarser scales that the options of a pyramid ask for.

    A writer that builds the scales of a pyramid while it writes them
    calls this function first, and then [`next_scale`][] for each
    coarser scale.

    Parameters
    ----------
    obj : Multiscale
        A multiscale object that holds its finest scale only.
    options : mapping
        The options of [`PYRAMID_OPTIONS`][], as
        [`pop_pyramid_options`][] took them out. `nscales=` takes the
        place of the argument `scales` of
        [`Multiscale.with_scales`][brainhops.datamodel.multiscale.Multiscale.with_scales],
        and gives a number of scales or their factors.

    Returns
    -------
    ScalePlan
        The factors, the mode and the alignment of the coarser scales.

    Raises
    ------
    TypeError
        If an option is not one of a pyramid, or if the options are
        wrong, as `with_scales` says.
    ValueError
        If the options are wrong, as `with_scales` says.
    """
    options = dict(options)
    unknown = sorted(set(options) - set(PYRAMID_OPTIONS))
    if unknown:
        raise TypeError(
            f"{', '.join(unknown)} are not options of a pyramid, which are "
            f"{', '.join(PYRAMID_OPTIONS)}."
        )
    scales = options.get("nscales")
    factor = options.get("factor")
    mode = options.get("mode", DEFAULT_MODE)
    align = options.get("align", DEFAULT_ALIGN)
    min_size = options.get("min_size", MIN_SIZE)
    strict = options.get("strict", False)
    if scales is not None and not isinstance(scales, Integral):
        if isinstance(scales, (str, bytes, Mapping)) or not isinstance(
            scales, Iterable
        ):
            raise TypeError(
                "The scales of a multiscale object are given by their "
                "number, by their factors or by their list, not by a "
                f"{type(scales).__name__}."
            )
        items = list(scales)
        if not items:
            raise ValueError(
                "A multiscale object has at least one scale, so the "
                "sequence of its scales or of its factors cannot be "
                "empty."
            )
        if factor is not None:
            raise TypeError(
                "`factor` gives the step of the rules, which a sequence "
                "of scales or of factors replaces."
            )
        if not all(is_factor(item) for item in items):
            raise TypeError(
                "The sequence of scales mixes factors and scales. Give "
                "the factors of the coarser scales, or the list of all "
                "the scales."
            )
        layout = pyramid_layout(obj)
        requested = _requested_factors(items, layout)
    else:
        if isinstance(scales, bool):
            raise TypeError("The number of scales is an int, not a bool.")
        layout = pyramid_layout(obj)
        step = _resolve_factor(
            2 if factor is None else factor, layout, layout.spatial
        )
        requested = default_factors(
            layout.shape,
            layout.sizes,
            step,
            None if scales is None else int(scales),
            min_size,
            strict,
            check_align(align),
        )
    check_mode(mode, field=layout.field, align=align)
    check_align(align)
    return ScalePlan(list(requested), mode, align)


def next_scale(obj: tx.Any, plan: ScalePlan) -> tx.Any:
    """Return a pyramid with the next coarser scale of a plan.

    A writer that builds the scales of a pyramid while it writes them
    gives this function the scales that it has written, read back from
    the store. The new scale is computed as
    [`Multiscale.with_scales`][brainhops.datamodel.multiscale.Multiscale.with_scales]
    computes it: from the scale before it when its factors are
    multiples of those of that scale, and from the finest scale
    otherwise, or always in the mode `"nearest"`. Its
    placement is that of the same scale in the pyramid that
    `with_scales` builds.

    Parameters
    ----------
    obj : Multiscale
        The scales already built, finest first.
    plan : ScalePlan
        The plan that [`plan_scales`][] returned.

    Returns
    -------
    Multiscale
        A copy of `obj` with one more scale.
    """
    count = obj.nscales
    return build_scales(
        obj, plan.factors[:count], plan.mode, plan.align, count - 1
    )


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


def is_factor(item: tx.Any) -> bool:
    """Return whether an item of a sequence of scales is a factor.

    A factor is an int, a mapping, or a sequence of ints. Any other
    item is a scale.
    """
    if isinstance(item, bool):
        return False
    if isinstance(item, (Integral, Mapping)):
        return True
    if isinstance(item, (str, bytes)) or not isinstance(item, (tuple, list)):
        return False
    return all(
        isinstance(value, Integral) and not isinstance(value, bool)
        for value in item
    )


def _requested_factors(
    items: tx.Sequence["FactorLike"], layout: PyramidLayout
) -> tx.List[tx.Tuple[int, ...]]:
    # Resolve the factors of the coarser scales, which never decrease.
    resolved = []  # type: tx.List[tx.Tuple[int, ...]]
    previous = (1,) * len(layout.shape)
    for index, entry in enumerate(items):
        level = _resolve_factor(
            entry, layout, layout.spatial, f"scales[{index}]"
        )
        decreasing = [
            position
            for position, (value, before) in enumerate(zip(level, previous))
            if value < before
        ]
        if decreasing:
            raise ValueError(
                f"The factors of the scales never decrease, but those of the "
                f"axes at positions {decreasing} decrease from "
                f"{list(previous)} to {list(level)}."
            )
        resolved.append(level)
        previous = level
    return resolved


def _resolve_factor(
    factor: "FactorLike",
    layout: PyramidLayout,
    default: tx.Sequence[int],
    name: str = "factor",
) -> tx.Tuple[int, ...]:
    """Return the factor of each axis of a level, from one of its forms.

    Names and positions are read by
    [`AxisList.positions`][brainhops.datamodel.axes.AxisList.positions],
    as
    [`Transformation.restrict`][brainhops.datamodel.transformations.Transformation.restrict]
    reads them, so that they designate the same axes as there.

    Parameters
    ----------
    factor : int or sequence of int or mapping
        The factor, in a form that `FactorLike` describes.
    layout : PyramidLayout
        The layout of the arrays, whose axes the factor refers to.
    default : sequence of int
        The positions of the axes that an int, and the key `None` of a
        mapping, apply to.
    name : str, default="factor"
        The name of the argument, for the messages of the errors.

    Returns
    -------
    tuple of int
        One factor per axis of the arrays.

    Raises
    ------
    TypeError
        If the factor has none of the three forms, or if a value is not
        an int.
    ValueError
        If a value is not positive, if a sequence has a length that
        matches neither the axes nor the axes downsampled by default, if
        a mapping names an axis twice, or if a factor above 1 is given
        to an axis that is never downsampled.
    KeyError
        If no axis has a name of a mapping.
    IndexError
        If a position of a mapping is out of range.
    """
    count = len(layout.shape)
    if isinstance(factor, bool):
        raise TypeError(f"`{name}` is an int, not a bool.")
    if isinstance(factor, Integral):
        values = [1] * count
        for position in default:
            values[position] = int(factor)
    elif isinstance(factor, Mapping):
        values = [1] * count
        seen = []  # type: tx.List[int]
        for key, value in factor.items():
            if key is None:
                continue
            for position in _positions(key, layout, name):
                if position in seen:
                    raise ValueError(
                        f"`{name}` gives a factor to the axis at position "
                        f"{position} more than once."
                    )
                seen.append(position)
                values[position] = value
        if None in factor:
            # The key `None` gives its factor to every axis that an int
            # would downsample and that the mapping does not name.
            fallback = factor[None]
            if isinstance(fallback, bool) or not isinstance(
                fallback, Integral
            ):
                raise TypeError(
                    f"`{name}` gives the factor {fallback!r} to the key "
                    "None, which is not an int."
                )
            for position in default:
                if position not in seen:
                    values[position] = fallback
    elif isinstance(factor, Iterable) and not isinstance(factor, (str, bytes)):
        given = list(factor)
        if len(given) == count:
            values = given
        elif len(given) == len(layout.spatial):
            values = [1] * count
            for position, value in zip(layout.spatial, given):
                values[position] = value
        else:
            raise ValueError(
                f"`{name}` gives {len(given)} factors, but the arrays have "
                f"{count} axes, of which {len(layout.spatial)} are "
                "downsampled by default."
            )
    else:
        raise TypeError(
            f"`{name}` is an int, a sequence of ints or a mapping from axes "
            f"to ints, not a {type(factor).__name__}."
        )
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(
                f"`{name}` gives the factor {value!r}, which is not an int."
            )
        if value < 1:
            raise ValueError(
                f"`{name}` gives the factor {value}, which is not positive."
            )
    for position in layout.fixed:
        if values[position] > 1:
            raise ValueError(
                f"`{name}` downsamples the axis at position {position}, "
                "which holds the components of the vectors of the field and "
                "is never downsampled."
            )
    return tuple(int(value) for value in values)


def _positions(
    reference: "AxisReference", layout: PyramidLayout, name: str
) -> tx.List[int]:
    # Return the non-negative positions that one reference designates.
    axes = layout.axes
    count = len(layout.shape)
    if isinstance(reference, Integral) and not -count <= reference < count:
        raise IndexError(
            f"`{name}` names the axis at position {reference}, which is out "
            f"of range for arrays of {count} axes."
        )
    try:
        positions = axes.positions([reference])
    except KeyError:
        names = [axis.name for axis in axes]
        raise KeyError(
            f"`{name}` names the axis {reference!r}, but the axes are named "
            f"{names}."
        ) from None
    if not positions:
        raise ValueError(f"`{name}` designates no axis with {reference!r}.")
    return [int(position) % count for position in positions]
