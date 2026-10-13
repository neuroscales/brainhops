"""Where the samples of an N5 dataset lie, from its spatial attributes.

The specification of N5 defines no spatial attribute. The functions of
this module read the attributes that the tools of the N5 ecosystem
write, as [`brainhops.io.common.n5`][brainhops.io.common.n5] describes
them with their sources, and give the voxel size, the downsampling
factors and the origin of one dataset. They do not build a
transformation, which the image readers do.
"""

# stdlib
import math
import warnings

# dependencies
import typing_extensions as tx

# internals
from brainhops.datamodel.units import Unit, is_physicalunit, is_spaceunit

# this format
from ._raw import N5Raw

_NO_UNIT = ("", "pixel", "pixels", "voxel", "voxels")
"""Units that n5-universe writes for sizes that have no physical unit."""


class N5AttributeWarning(UserWarning):
    """A spatial attribute of an N5 node was ignored.

    The spatial attributes of N5 are conventions of its tools, not part
    of its specification, so a value that breaks a convention, such as a
    `pixelResolution` of the wrong length or a unit that brainhops does
    not know, is ignored with this warning instead of making the dataset
    unreadable.
    """


class Placement(tx.NamedTuple):
    """The voxel size, the downsampling and the origin of one dataset.

    The values are those that the function `level_transformation` of
    the raster image readers takes: the voxel `i` of the dataset along
    an axis lies at
    `size * (factor * i + (factor - 1) / 2) + origin`. Each tuple lists
    the leading axes of the dataset, fastest first, and may be shorter
    than the number of dimensions, in which case the other axes have a
    size and a factor of 1 and an origin of 0.
    """

    sizes: tx.Tuple[float, ...]
    """The size of a voxel of the finest level, or `()` when unknown."""

    unit: tx.Optional[str]
    """The unit of the sizes, or `None` when they have no unit."""

    factors: tx.Tuple[float, ...]
    """The downsampling factors of the dataset from the finest level."""

    origin: tx.Tuple[float, ...]
    """The position of the centre of the first voxel of the finest level."""


def placement(record: N5Raw, level: tx.Optional[int] = None) -> Placement:
    """Return the placement of a dataset from the spatial attributes.

    For a record of a dataset, only its own attributes are read, as the
    n5-universe library does. For a record of a pyramid, the dataset is
    the level `level` (0 by default), and an attribute that the dataset
    does not give is taken from the group of the pyramid, as the
    deprecated n5-viewer layout and Paintera store it.

    Parameters
    ----------
    record : N5Raw
        The record of a dataset or of a pyramid.
    level : int, optional
        The index of the level of a pyramid, which is ignored for a
        dataset.

    Returns
    -------
    Placement
        The placement of the dataset.
    """
    if record.levels:
        index = 0 if level is None else level
        dataset, group = record.levels[index], record.attributes
    else:
        index, dataset, group = 0, record, {}
    ndim = len(dataset.dimensions or ())
    own = dataset.attributes

    sizes, unit = _resolution(own, ndim)
    from_group = False
    if sizes is None:
        sizes, unit = _resolution(group, ndim)
        from_group = sizes is not None
    factors = _numbers(own, "downsamplingFactors", ndim, positive=True)
    if factors is None:
        factors = _group_scale(group, index, ndim)
    factors = factors or ()

    offset = _numbers(own, "offset", ndim)
    if offset is not None and not from_group:
        # A dataset that gives its own `offset` gives the position of its
        # first voxel, whatever its downsampling.
        shift = _shift(sizes or (), factors, len(offset))
        origin = tuple(o - s for o, s in zip(offset, shift))
    else:
        # The `offset` of a group is the position of the first voxel of
        # the finest level, from which the levels are shifted.
        origin = _numbers(group, "offset", ndim) or ()
    return Placement(
        sizes=tuple(sizes or ()),
        unit=unit,
        factors=tuple(factors),
        origin=tuple(origin),
    )


# ----------------------------------------------------------------------
#   ATTRIBUTES
# ----------------------------------------------------------------------


def _resolution(
    attributes: tx.Mapping[str, tx.Any], ndim: int
) -> tx.Tuple[tx.Optional[tx.Tuple[float, ...]], tx.Optional[str]]:
    """Return the voxel size of a node and its unit, or `(None, None)`.

    `pixelResolution` is read first, as an object with `dimensions` and
    `unit` or as a bare array, which has no unit. `resolution` is read
    next, with the unit of the `unit` attribute.
    """
    value = attributes.get("pixelResolution")
    if isinstance(value, dict):
        sizes = _numbers(
            {"pixelResolution": value.get("dimensions")},
            "pixelResolution",
            ndim,
            positive=True,
        )
        if sizes is not None:
            return sizes, _unit(value.get("unit"))
    elif value is not None:
        sizes = _numbers(attributes, "pixelResolution", ndim, positive=True)
        if sizes is not None:
            return sizes, None
    sizes = _numbers(attributes, "resolution", ndim, positive=True)
    if sizes is not None:
        return sizes, _unit(attributes.get("unit"))
    return None, None


def _group_scale(
    group: tx.Mapping[str, tx.Any], index: int, ndim: int
) -> tx.Optional[tx.Tuple[float, ...]]:
    """Return the factors of a level from the `scales` of its group.

    The deprecated layout of the n5-viewer lists the downsampling factors
    of every level, finest first, in the `scales` of the group.
    """
    scales = group.get("scales")
    if scales is None:
        return None
    if not isinstance(scales, list) or index >= len(scales):
        _ignored("scales", scales)
        return None
    return _numbers(
        {"scales": scales[index]}, "scales", ndim, positive=True
    )


def _numbers(
    attributes: tx.Mapping[str, tx.Any],
    name: str,
    ndim: int,
    positive: bool = False,
) -> tx.Optional[tx.Tuple[float, ...]]:
    """Return an attribute that is an array of at most `ndim` numbers.

    A missing attribute gives `None`. An attribute that is not such an
    array, or whose values are not finite, or not positive when
    `positive` is true, is ignored with an `N5AttributeWarning`.
    """
    value = attributes.get(name)
    if value is None:
        return None
    if (
        not isinstance(value, list)
        or not 0 < len(value) <= ndim
        or any(
            isinstance(v, bool) or not isinstance(v, (int, float))
            for v in value
        )
        or any(not math.isfinite(v) for v in value)
        or (positive and any(v <= 0 for v in value))
    ):
        _ignored(name, value)
        return None
    return tuple(float(v) for v in value)


def _unit(value: tx.Any) -> tx.Optional[str]:
    """Return a length unit that brainhops knows, or `None`.

    A unit that n5-universe uses for sizes without a physical unit, such
    as `"pixel"`, gives `None`. An unknown unit, or a unit that is not a
    length, gives `None` with an `N5AttributeWarning`.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        _ignored("unit", value)
        return None
    if value.strip().lower() in _NO_UNIT:
        return None
    try:
        unit = Unit(value)
    except Exception:  # noqa: BLE001
        unit = None
    if unit is None or not (is_spaceunit(unit) and is_physicalunit(unit)):
        _ignored("unit", value)
        return None
    return value


def _shift(
    sizes: tx.Sequence[float], factors: tx.Sequence[float], count: int
) -> tx.Tuple[float, ...]:
    """Return the shift `size * (factor - 1) / 2` of the leading axes."""
    out = []
    for i in range(count):
        size = sizes[i] if i < len(sizes) else 1.0
        factor = factors[i] if i < len(factors) else 1.0
        out.append(size * (factor - 1) / 2)
    return tuple(out)


def _ignored(name: str, value: tx.Any) -> None:
    """Warn that a spatial attribute is ignored."""
    warnings.warn(
        f"The N5 attribute {name!r} is {value!r}, which brainhops does not "
        f"understand, so it is ignored.",
        N5AttributeWarning,
        stacklevel=3,
    )
