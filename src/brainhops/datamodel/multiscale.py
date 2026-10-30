"""Pyramids of scales, and the method that builds their coarser scales.

A multiscale object, such as a
[`MultiScaleImage`][brainhops.datamodel.images.MultiScaleImage] or a
[`MultiscaleField`][brainhops.datamodel.transformations.MultiscaleField],
holds the same image or field at several resolutions, finest first, so
that a viewer or an algorithm that only needs a coarse view reads less
data. [`Multiscale.with_scales`][] gives it the scales that it should
have: as many as the rules allow, a number of them, the factors of each
of them, or a complete list of them. The arithmetic, its modes, its
alignments and its edges are explained in
[`brainhops._core.pyramids`][].

A conversion builds the scales too, with the same values under the
keyword `nscales=` and the options of `with_scales`, as
[`PYRAMID_OPTIONS`][] lists them: `image.to(MultiScaleImage, nscales=4)`
calls `with_scales(4)` on the converted pyramid. A class whose
[`BUILDS_SCALES`][Multiscale.BUILDS_SCALES] is true, such as the
formats of OME-Zarr, builds them even without `nscales=` when it is
converted from a single scale, as `with_scales()` builds them, so that
`image.to(OmeZarrImage)` and `io.save(image, "image.ome.zarr")` write a
full pyramid, and `nscales=1` keeps the single scale.
"""

# stdlib
from collections.abc import Iterable, Mapping
from numbers import Integral

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import NotKwOnly

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
from .axes import AxisList, AxisQuery
from .base import DataModelBase

SINGLE_SCALE = tx.TypeVar("SINGLE_SCALE", covariant=True)

AxisReference: tx.TypeAlias = tx.Union[int, AxisQuery]
"""One axis of a pyramid: its position, its name or another axis query."""

FactorLike: tx.TypeAlias = tx.Union[
    int, tx.Sequence[int], tx.Mapping[tx.Optional[AxisReference], int]
]
"""The factor of each axis of one level, in one of three forms.

An int is the factor of every axis that is downsampled by default, which
are the spatial axes, and the other axes get 1. A sequence gives one
factor per axis of the arrays, or one factor per axis that is
downsampled by default, in their order. A mapping gives the factor of
the axes that it names, by position, which counts from the end when it
is negative, or by name, and the other axes get 1, unless the mapping
has the key `None`. The factor of the key `None` goes to every axis that
an int would downsample and that the mapping does not name, so that
`{"z": 1, None: 2}` halves the spatial axes but `z`. It never reaches an
axis that an int leaves alone, such as a time axis or the axis of the
vectors of a field, which a mapping must name to downsample. A factor of
1 leaves its axis as it is.
"""

PYRAMID_OPTIONS = ("nscales", "factor", "mode", "align", "min_size", "strict")
"""The options of a conversion that build the scales of a multiscale object.

`nscales=` takes the values of the argument `scales` of
[`Multiscale.with_scales`][], and the other options are those of that
method.
"""


class PyramidLayout(tx.NamedTuple):
    """What a multiscale object tells about the arrays of its finest scale.

    The method [`Multiscale.with_scales`][] resolves the factors that it
    is given against this layout.
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


class Multiscale(DataModelBase, tx.Generic[SINGLE_SCALE]):
    """Mixin for a pyramid of scales, finest first.

    Index 0 is the scale with the highest resolution.

    This base class provides the list of scales, the operations that
    select one of these scales, and the method that gives the pyramid
    its scales, [`with_scales`][].

    It does not define what a scale is: a subclass supplies the scales
    and, through the `_level_resolution` hook, the physical grid size of
    each one, from which `nearest_level` picks the scale whose
    resolution is closest to a target grid. A subclass that builds its
    scales also supplies the hooks `_pyramid_layout`, `_keeps_scale`,
    `_build_scales`, `_check_scales` and `_adopt_scales`.
    """

    BUILDS_SCALES: tx.ClassVar[bool] = False
    """Whether converting a single scale builds the coarser scales.

    When it is true, a conversion to this class from a single-scale
    object builds as many scales as `with_scales()` builds, unless
    `nscales=` says otherwise. The formats of OME-Zarr set it.
    """

    # --- attributes ---------------------------------------------------

    scales: NotKwOnly[tx.List[SINGLE_SCALE]] = ()
    """Scales, from finest to coarsest."""

    @property
    def nscales(self) -> int:
        """Number of scales."""
        return len(self.scales)

    @property
    def _finest(self) -> SINGLE_SCALE:
        # Return the finest scale, or `None` if there are no scales.
        scales = self.scales
        return scales[0] if scales else None

    # --- methods ------------------------------------------------------

    def to_singlescale(self, index: int = 0) -> SINGLE_SCALE:
        """Return the scale at `index`, where 0 is the finest.

        The scale is returned as it is stored: for a field, it is the plain
        transformation of that scale, not a multiscale field.
        """
        return self.scales[int(index)]

    @tx.overload
    def with_scales(
        self,
        scales: None = None,
        *,
        factor: tx.Optional[FactorLike] = None,
        mode: str = DEFAULT_MODE,
        align: str = DEFAULT_ALIGN,
        min_size: int = MIN_SIZE,
        strict: bool = False,
    ) -> tx.Self: ...

    @tx.overload
    def with_scales(
        self,
        scales: int,
        *,
        factor: tx.Optional[FactorLike] = None,
        mode: str = DEFAULT_MODE,
        align: str = DEFAULT_ALIGN,
        min_size: int = MIN_SIZE,
        strict: bool = False,
    ) -> tx.Self: ...

    @tx.overload
    def with_scales(
        self,
        scales: tx.Sequence[FactorLike],
        *,
        mode: str = DEFAULT_MODE,
        align: str = DEFAULT_ALIGN,
    ) -> tx.Self: ...

    @tx.overload
    def with_scales(self, scales: tx.Sequence[SINGLE_SCALE]) -> tx.Self: ...

    def with_scales(
        self,
        scales: tx.Union[
            None, int, tx.Sequence[FactorLike], tx.Sequence[SINGLE_SCALE]
        ] = None,
        *,
        factor: tx.Optional[FactorLike] = None,
        mode: str = DEFAULT_MODE,
        align: str = DEFAULT_ALIGN,
        min_size: int = MIN_SIZE,
        strict: bool = False,
    ) -> tx.Self:
        """Return a copy with the given scales, finest first.

        A pyramid lets a viewer that shows a large image whole, or an
        algorithm that works from coarse to fine, read a coarse scale
        instead of the finest one. This method gives a multiscale object
        the scales that it should have. The finest scale is always kept.

        `scales` is read by its type:

        - `None`, the default, builds as many scales as the rules allow.
        - An int is the number of scales, the finest one included. When
          the rules stop before, fewer scales are built, unless `strict`
          is true.
        - A sequence of factors gives, for each coarser scale, its
          factor from the finest scale along each axis, in a form that
          `FactorLike` describes. The factors are positive integers, and
          the factor of an axis never decreases from a scale to the
          next.
        - A sequence of scales, such as single-scale images for an
          image, is the complete list of scales, finest first, which the
          copy takes as they are. They must be ordered from fine to
          coarse and share one world space.

        The rules build the scales step by step. At each step, an axis
        is downsampled by its factor of `factor`, which is 2 for the
        spatial axes and 1, which leaves an axis as it is, for the other
        axes by default. An axis is downsampled while it has more than
        `min_size` voxels, even when its next scale has fewer, so that
        an axis of 40 voxels becomes an axis of 20 voxels and an axis of
        32 voxels stays as it is. The rules stop when no axis can be
        downsampled. An axis whose voxels are more than `sqrt(f)` times
        larger than the smallest voxels of the axes that can still be
        downsampled waits, so that anisotropic voxels become roughly
        isotropic first.

        A scale is computed from the scale before it whenever its
        factors are multiples of the factors of that scale, and from the
        finest scale otherwise, or always in the mode `"nearest"`, which
        picks the voxels of the finest scale. A coarser scale that the
        object already holds is kept when its factors, its placement and
        its shape are those that are asked for, and the scales after it
        are built from it. The voxels of a scale lie where `align` says,
        and its values are computed as `mode` says, as
        [`brainhops._core.pyramids`][] explains.

        !!! example
            ```python
            pyramid.with_scales()
            pyramid.with_scales(4, factor={"z": 1, None: 2})
            pyramid.with_scales([2, 4, {"x": 8, "y": 8, "z": 4}])
            pyramid.with_scales(3, mode="mode")  # labels
            ```

        Parameters
        ----------
        scales : int or sequence, optional
            The number of scales, the factors of the coarser scales, or
            the list of scales. By default, as many scales as the rules
            allow.
        factor : int or sequence of int or mapping, optional
            The factor of each axis at each step of the rules, in a form
            that `FactorLike` describes. By default, 2 for the spatial
            axes. It is refused with a sequence of `scales`.
        mode : str, default="mean"
            How a coarse voxel is computed from the voxels that it
            covers, one of [`MODES`][brainhops._core.pyramids.MODES].
        align : {"exact", "edges", "centers"}, default="exact"
            How the voxels of a coarse scale are laid over those of the
            finest scale. `"exact"` crops each axis to a multiple of `f`
            and puts the voxel `i` at the voxel `f * i + (f - 1) / 2`,
            which is the rule of OME-NGFF 0.6. `"edges"` keeps the outer
            edges of the field of view, with `ceil(n / f)` voxels, and
            `"centers"` keeps the centres of the first and the last
            voxels, and both pull the voxels by interpolation after
            smoothing them. The modes `"mode"`, `"max"` and `"min"`
            take `"exact"` only.
        min_size : int, default=32
            The number of voxels at or below which the rules stop
            downsampling an axis.
        strict : bool, default=False
            Whether to refuse to build fewer scales than an int `scales`
            asks for.

        Returns
        -------
        Self
            The copy with its scales.

        Raises
        ------
        TypeError
            If `scales` is none of these types, mixes factors and
            scales, or comes with `factor`, or if this class cannot
            build its scales.
        ValueError
            If `scales` is empty, if `strict` is true and the rules stop
            early, if the factors are not positive or decrease, if a
            list of scales is not ordered from fine to coarse or does
            not share one world space, if a factor downsamples the axis
            of the vectors of a field, or if the mode or the alignment
            does not apply.
        KeyError
            If no axis has a name that a factor gives.
        IndexError
            If a position that a factor gives is out of range.
        """
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
            factors = [_is_factor(item) for item in items]
            if not any(factors):
                self._check_scales(items)
                return self._adopt_scales(items)
            if not all(factors):
                raise TypeError(
                    "The sequence of scales mixes factors and scales. Give "
                    "the factors of the coarser scales, or the list of all "
                    "the scales."
                )
            layout = self._pyramid_layout()
            requested = _requested_factors(items, layout)
        else:
            if isinstance(scales, bool):
                raise TypeError("The number of scales is an int, not a bool.")
            layout = self._pyramid_layout()
            step = resolve_factor(
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
        kept = 0
        for index, level in enumerate(requested, start=1):
            if index >= self.nscales:
                break
            if not self._keeps_scale(index, level, align):
                break
            kept = index
        return self._build_scales(requested, mode, align, kept)

    # --- hooks --------------------------------------------------------

    def _level_resolution(self, index: int) -> tx.Optional[ArrayProtocol]:
        # Return the physical grid size of a level, as a vector with one entry
        # per axis in the input units, or `None` if the size is unknown. This
        # hook returns `None` here, and subclasses override it.
        return None

    def _pyramid_layout(self) -> PyramidLayout:
        # Return the layout of the arrays of the finest scale. A
        # subclass that builds its scales overrides it.
        raise TypeError(
            f"A {type(self).__name__} cannot build its scales from its finest "
            "scale."
        )

    def _keeps_scale(
        self, index: int, factors: tx.Tuple[int, ...], align: str
    ) -> bool:
        # Return whether the scale at `index` lies where `align` puts a
        # scale of `factors` from the finest scale, and has its shape.
        return False

    def _build_scales(
        self,
        factors: tx.Sequence[tx.Tuple[int, ...]],
        mode: str,
        align: str,
        kept: int,
    ) -> tx.Self:
        # Return a copy that keeps the scales up to `kept` and builds
        # the others of `factors`, one tuple of factors per axis of the
        # layout and per coarser scale, from the scale `kept` or from
        # the finest scale.
        raise TypeError(
            f"A {type(self).__name__} cannot build its scales from its finest "
            "scale."
        )

    def _check_scales(self, scales: tx.Sequence[SINGLE_SCALE]) -> None:
        # Refuse a list of scales that is not ordered from fine to
        # coarse, or whose scales do not share one world space.
        return None

    def _adopt_scales(self, scales: tx.Sequence[SINGLE_SCALE]) -> tx.Self:
        # Return a copy that holds `scales` as they are.
        raise TypeError(
            f"A {type(self).__name__} cannot be given a list of scales."
        )


def _is_factor(item: tx.Any) -> bool:
    # Whether an item of a sequence of scales is a factor rather than a
    # scale: an int, a mapping, or a sequence of ints.
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
    items: tx.Sequence[FactorLike], layout: PyramidLayout
) -> tx.List[tx.Tuple[int, ...]]:
    # Resolve the factors of the coarser scales, which never decrease.
    resolved = []  # type: tx.List[tx.Tuple[int, ...]]
    previous = (1,) * len(layout.shape)
    for index, entry in enumerate(items):
        level = resolve_factor(
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


def resolve_factor(
    factor: FactorLike,
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
    reference: AxisReference, layout: PyramidLayout, name: str
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
        [`BUILDS_SCALES`][Multiscale.BUILDS_SCALES] is true does.

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
