"""
Conventions shared by every headerless raster image format.

A raster format -- PNG, JPEG, BMP, GIF, WebP, TIFF and their kin -- stores
an array of samples and, at most, a pixel size. It records no origin, no
orientation and no world space. This module holds the conventions such a
format is read and written with, so that every raster reader (the Pillow
reader, the TIFF reader, ...) agrees on them. It mirrors
[`brainhops.io.common.nifti`][], which holds what the NIfTI-based images and
transformations share. Nothing in it depends on a particular decoding
library: a backend (Pillow, tifffile) decodes the file into a C-ordered
array and a description of its axes, and this module does the rest.

## Conventions

1. **Index space.** Coordinates are 0-based and an integer coordinate is
   the *centre* of a pixel, as in NIfTI and ITK. The pixel `(0, 0)` covers
   `[-0.5, 0.5] x [-0.5, 0.5]` in index space.
2. **Axis order.** A decoding library returns a C-ordered array, whose
   last axis changes fastest -- `(rows, columns)`, or `(y, x)`, for a
   plain image. Readers return a *view* of that array transposed into the
   brainhops order (`to_canonical`)
   -- the spatial axes first, fastest first (`x, y[, z]`), then time, then
   the channel axis, then anything else -- without copying it, as nibabel
   does for NIfTI. Writers transpose back
   (`to_storage`).
3. **Row 0 is at the top.** The first row of a raster file is the top of
   the picture, so `y` points *down*. This is documented, not encoded: the
   reader attaches no orientation to its axes.
4. **Channels.** Colour (or any per-pixel sample) components are put on a
   [`ChannelAxis`][brainhops.datamodel.axes.ChannelAxis] named `"c"`,
   after the spatial axes. A single-component image has no channel axis.
5. **Pixel size.** The voxel-to-world transformation is a single
   [`Scaling`][brainhops.datamodel.transformations.Scaling] from the
   pixel (or voxel) system, whose axes count samples, to a system named
   `"physical"`. When the format records a meaningful pixel size the
   scaling carries it, and the physical axes carry its unit.
6. **Unknown size.** When the size is absent, or is one of the
   placeholders writers put in when they know nothing (72 or 96 dpi, or
   tifffile's 1 dpi), it is *unknown*: the scaling is the identity and the
   physical axes have no unit (`None`, "unspecified"). A physical size is
   never invented.
7. **Overrides.** Every reader takes the keywords `pixel_size` and `unit`
   (`resolve_pixel_size`),
   which take precedence over the file's metadata.

## Building a reader on this module

A reader decodes its file into a C-ordered array, the storage axes of that
array (a string of single-letter codes such as `"YX"`, `"YXS"` or
`"TZCYX"`, as tifffile spells them, or a list of
[`Axis`][brainhops.datamodel.axes.Axis]), and whatever pixel size its
metadata records, as a mapping from axis name to `(size, unit)`. Then:

```python
axes = storage_axes("ZYXS")                    # C order
data, axes = to_canonical(array, axes)         # F-ordered view: x, y, z, c
scales = resolve_pixel_size(axes, metadata, pixel_size=..., unit=...)
transformations = raster_transformations(axes, scales)
```

A reader of a richer format adds what it knows to `scales`: a time
interval on the `"t"` axis, say. Each level of a multiscale pyramid is
built the same way, with its own sizes.

A writer does the converse: it finds the canonical axes of the image
(`image_axes`), transposes the data
into its storage order
(`to_storage`) and reads the pixel
size back from the preferred transformation
(`physical_pixel_size`),
which is `None` when the size is unknown or the transformation is more
than a scaling.
"""

__all__ = [
    "MM_PER_INCH",
    "DEFAULT_DPIS",
    "AxisScale",
    "storage_axes",
    "axis_group",
    "spatial_axes",
    "canonical_permutation",
    "to_canonical",
    "storage_permutation",
    "to_storage",
    "default_axes",
    "image_axes",
    "pixel_system",
    "physical_system",
    "resolve_pixel_size",
    "raster_transformations",
    "level_transformation",
    "physical_pixel_size",
    "convert_length",
    "is_default_dpi",
    "dpi_to_size",
    "size_to_dpi",
]

# stdlib
import math
from numbers import Number

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import replace

# internals
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    Scaling,
    Sequence,
    Transformation,
)
from brainhops.datamodel.units import (
    Unit,
    is_physicalunit,
    is_spaceunit,
)

# ----------------------------------------------------------------------
#   CONSTANTS
# ----------------------------------------------------------------------

MM_PER_INCH: float = 25.4
"""The number of millimetres in an inch, which converts a resolution in
dots per inch to a pixel size: `size_mm = 25.4 / dpi`."""

DEFAULT_DPIS: tx.Tuple[float, ...] = (1.0, 72.0, 96.0)
"""
Resolutions that say nothing about the physical size of a pixel.

Writers that know nothing about the pixel size still have to fill the
resolution fields of some formats, and put in a placeholder: 72 dpi (the
historical screen resolution, and the default of many editors), 96 dpi
(Windows' screen resolution) or 1 (tifffile's default, which is "one pixel
per unit"). A resolution equal to one of these is read as *unknown*.
"""

_DPI_TOLERANCE = 1e-3
# PNG stores pixels per metre as an integer, so 72 dpi comes back as
# 72.009 (2835 pixels per metre). A resolution this close to a placeholder,
# relatively, is the placeholder.

_INDEX = "index"
# The unit of an axis that counts samples, i.e. an array axis.

AxisScale = tx.Tuple[float, tx.Optional[tx.Union[str, Unit]]]
"""
The size of one sample along an axis, and the unit it is measured in.

The unit is `None` when it is unspecified, in which case the size has no
physical meaning.
"""

_AxesLike = tx.Union[str, tx.Sequence[Axis]]

# Single-letter storage codes, as tifffile spells them, and the axis each
# one stands for: its name and its type. A code that is not listed is an
# axis of unknown type, named after the code.
_CODE_ROLES: tx.Dict[str, tx.Tuple[str, tx.Optional[str]]] = {
    "X": ("x", "space"),
    "Y": ("y", "space"),
    "Z": ("z", "space"),
    "T": ("t", "time"),
    "C": ("c", "channel"),
    "S": ("c", "channel"),  # "samples per pixel", e.g. RGB components
}

# The canonical (brainhops) order of axis groups, as in NIfTI and in the
# Zarr reader: space, then time, then channels, then the rest.
_GROUPS = ("space", "time", "channel", "other")

# The position of a named spatial axis within the spatial group.
_SPACE_RANK = {"x": 0, "y": 1, "z": 2}

# A channel-like axis: colour components, or the components of a vector.
_CHANNEL_TYPES = ("channel", "displacement", "coordinate")


# ----------------------------------------------------------------------
#   AXES
# ----------------------------------------------------------------------


def storage_axes(codes: _AxesLike) -> tx.List[Axis]:
    """
    The axes of a C-ordered array, from their storage codes.

    Each character of `codes` names one axis of the array, slowest first,
    as tifffile spells them: `X`, `Y` and `Z` are spatial axes, `T` is
    time, `C` (channels) and `S` (samples per pixel, such as the RGB
    components of a pixel) are channel axes. Any other letter is an axis
    of unknown type, named after the letter in lower case. Every axis
    counts samples: its unit is the index unit.

    The channel axis is named `"c"`. When a file has both a `C` and an
    `S` axis, the `S` axis is named `"s"`. A name that would repeat is
    numbered (`"q"`, `"q1"`, ...).

    A list of [`Axis`][brainhops.datamodel.axes.Axis] is returned as a
    list, unchanged.

    Parameters
    ----------
    codes : str | Sequence[Axis]
        The storage codes, such as `"YX"`, `"YXS"` or `"TZCYX"`.

    Returns
    -------
    list[Axis]
        One axis per dimension, in the storage (C) order.
    """
    if not isinstance(codes, str):
        axes = list(codes)
        for axis in axes:
            if not isinstance(axis, Axis):
                raise TypeError(
                    f"Expected storage codes or a list of Axis, got an "
                    f"element of type {type(axis).__name__}."
                )
        return axes
    upper = codes.upper()
    axes: tx.List[Axis] = []
    seen: tx.Dict[str, int] = {}
    for code in upper:
        name, type_ = _CODE_ROLES.get(code, (code.lower(), None))
        if code == "S" and "C" in upper:
            name = "s"
        count = seen.get(name, 0)
        seen[name] = count + 1
        if count:
            name = f"{name}{count}"
        axes.append(Axis(name, type_, unit=_INDEX))
    return axes


def axis_group(axis: Axis) -> str:
    """
    The group an axis is sorted into: `"space"`, `"time"`, `"channel"` or
    `"other"`.

    A displacement or coordinate axis, which holds the components of a
    vector, is grouped with the channels.
    """
    type_ = getattr(axis, "type", None)
    if type_ in _CHANNEL_TYPES:
        return "channel"
    if type_ in ("space", "time"):
        return type_
    return "other"


def spatial_axes(axes: tx.Sequence[Axis]) -> tx.List[Axis]:
    """The spatial axes among `axes`, in order."""
    return [axis for axis in axes if axis_group(axis) == "space"]


def canonical_permutation(axes: _AxesLike) -> tx.List[int]:
    """
    The permutation that reorders storage axes into the brainhops order.

    The brainhops order lists the spatial axes first (`x`, `y`, `z`), then
    time, then the channel axes, then the others. Within a group, axes
    are listed fastest first, i.e. in the reverse of their storage order,
    except that named spatial axes always sort into `x, y, z`. So a
    `"ZYX"` array is simply reversed, and a `"YXS"` (RGB) array becomes
    `x, y, c`.

    Parameters
    ----------
    axes : str | Sequence[Axis]
        The storage axes, slowest first.

    Returns
    -------
    list[int]
        Element `i` is the storage position of the axis that becomes axis
        `i` in the brainhops order. The same permutation reorders the
        array (`array.transpose(perm)`) and the list of axes.
    """
    axes = storage_axes(axes)
    n = len(axes)

    def key(position: int) -> tx.Tuple[int, int, int]:
        axis = axes[position]
        group = axis_group(axis)
        rank = 0
        if group == "space":
            rank = _SPACE_RANK.get(getattr(axis, "name", None), 3)
        return (_GROUPS.index(group), rank, n - 1 - position)

    return sorted(range(n), key=key)


def to_canonical(
    array: ArrayProtocol, axes: _AxesLike
) -> tx.Tuple[ArrayProtocol, tx.List[Axis]]:
    """
    Transpose a C-ordered array into the brainhops order, without a copy.

    Parameters
    ----------
    array : ArrayProtocol
        The array, as the decoding library returns it.
    axes : str | Sequence[Axis]
        Its storage axes, slowest first (see
        `storage_axes`).

    Returns
    -------
    data : ArrayProtocol
        A transposed view of `array`. A plain (`"YX"`) or volumetric
        (`"ZYX"`) image comes back exactly F-ordered.
    axes : list[Axis]
        The axes of `data`, in order.
    """
    axes = storage_axes(axes)
    if len(axes) != len(array.shape):
        raise ValueError(
            f"An array of {len(array.shape)} dimensions cannot be "
            f"described by {len(axes)} axes."
        )
    perm = canonical_permutation(axes)
    return array.transpose(perm), [axes[p] for p in perm]


def storage_permutation(
    axes: tx.Sequence[Axis], codes: _AxesLike
) -> tx.List[int]:
    """
    The permutation that reorders brainhops axes into a storage order.

    Each storage axis is matched to an axis of `axes` with the same name
    or, failing that, to the first one left of the same type (so an
    unnamed spatial axis still matches `X`). This is the converse of
    `canonical_permutation`
    for a writer that has a fixed storage order, such as `"YXS"`.

    Parameters
    ----------
    axes : Sequence[Axis]
        The axes of the image, in the brainhops order.
    codes : str | Sequence[Axis]
        The storage axes the file wants, slowest first.

    Returns
    -------
    list[int]
        Element `i` is the position, in `axes`, of the axis stored at
        position `i`.

    Raises
    ------
    ValueError
        If the two do not have the same number of axes, or an axis cannot
        be matched.
    """
    targets = storage_axes(codes)
    if len(targets) != len(axes):
        raise ValueError(
            f"Cannot store {len(axes)} axes as {len(targets)} axes."
        )
    names = [getattr(axis, "name", None) for axis in axes]
    space = [i for i, axis in enumerate(axes) if axis_group(axis) == "space"]
    perm: tx.List[tx.Optional[int]] = [None] * len(targets)
    left = list(range(len(axes)))
    # First, match by name.
    for j, target in enumerate(targets):
        name = getattr(target, "name", None)
        if name is not None and name in names and names.index(name) in left:
            perm[j] = names.index(name)
            left.remove(perm[j])
    # Then, a spatial axis named x, y or z is the first, second or third
    # spatial axis of the image, and anything else is the first axis left
    # of the same group.
    for j, target in enumerate(targets):
        if perm[j] is not None:
            continue
        group = axis_group(target)
        rank = _SPACE_RANK.get(getattr(target, "name", None))
        match = None
        if group == "space" and rank is not None and rank < len(space):
            if space[rank] in left:
                match = space[rank]
        if match is None:
            match = next(
                (i for i in left if axis_group(axes[i]) == group), None
            )
        if match is None:
            raise ValueError(
                f"No axis of the image matches the storage axis "
                f"{getattr(target, 'name', None)!r}."
            )
        perm[j] = match
        left.remove(match)
    return perm


def to_storage(
    array: ArrayProtocol, axes: tx.Sequence[Axis], codes: _AxesLike
) -> ArrayProtocol:
    """
    Transpose an image's data from the brainhops order into a storage
    order.

    The result is a view. Pass it to `numpy.ascontiguousarray` if the
    writer needs C-contiguous memory.
    """
    return array.transpose(storage_permutation(axes, codes))


def default_axes(ndim: int, channel: bool = False) -> tx.List[Axis]:
    """
    The axes to assume for an array that comes with no description.

    The spatial axes `x, y, z` come first (as many as there are, up to
    three), then time, then a channel axis, then unknown axes. With
    `channel=True`, the last axis is a channel axis and the others are
    spatial (up to three) or unknown.
    """
    if channel and ndim >= 1:
        others = default_axes(ndim - 1)
        return others + [Axis("c", "channel", unit=_INDEX)]
    names = [("x", "space"), ("y", "space"), ("z", "space")]
    names += [("t", "time"), ("c", "channel")]
    axes = [
        Axis(*names[i], unit=_INDEX)
        if i < len(names)
        else Axis(f"dim{i}", unit=_INDEX)
        for i in range(ndim)
    ]
    return axes


def image_axes(
    ndim: int, system: tx.Optional[CoordinateSystem] = None
) -> tx.Optional[tx.List[Axis]]:
    """
    The axes of an image's data, in the brainhops order, if they are known.

    They are the axes of the input system of the image's preferred
    transformation, when that system lists exactly one axis per dimension.
    Otherwise, they are not known, and `None` is returned so that the
    writer can apply its own default (see
    `default_axes`).
    """
    axes = getattr(system, "axes", None)
    if axes is None:
        return None
    try:
        axes = list(axes)
    except TypeError:
        return None
    if len(axes) != ndim or any(not isinstance(a, Axis) for a in axes):
        return None
    return axes


# ----------------------------------------------------------------------
#   COORDINATE SYSTEMS
# ----------------------------------------------------------------------


def pixel_system(axes: tx.Sequence[Axis]) -> CoordinateSystem:
    """
    The coordinate system of a raster's index space.

    Its axes are `axes`, each counting samples, listed F-ordered (fastest
    first). It is named `"pixel"` when the raster has two spatial axes, and
    `"voxel"` otherwise, as a NIfTI image's index space is.
    """
    axes = [replace(axis, unit=_INDEX) for axis in axes]
    name = "pixel" if len(spatial_axes(axes)) == 2 else "voxel"
    return CoordinateSystem(name=name, axes=axes, order="F")


def physical_system(
    axes: tx.Sequence[Axis],
    units: tx.Optional[tx.Mapping[str, tx.Any]] = None,
) -> CoordinateSystem:
    """
    The coordinate system a raster's pixel size maps its index space to.

    It is named `"physical"`, and has the same axes as the index space,
    each measured in the unit `units` gives for its name, or in no unit at
    all (`None`, "unspecified") when it gives none. An axis is never left
    counting samples: the system measures, it does not index.
    """
    units = dict(units or {})
    out = [replace(axis, unit=units.get(axis.name)) for axis in axes]
    return CoordinateSystem(name="physical", axes=out)


# ----------------------------------------------------------------------
#   PIXEL SIZE
# ----------------------------------------------------------------------


def _meters(unit: tx.Any) -> tx.Optional[float]:
    """The length of a unit of space in metres, or `None`."""
    if unit is None:
        return None
    if not isinstance(unit, Unit):
        try:
            unit = Unit(unit)
        except Exception:
            return None
    if not (is_spaceunit(unit) and is_physicalunit(unit)):
        return None
    scale = getattr(unit, "scale", None)
    if isinstance(scale, Number) and scale > 0:
        return float(scale)
    return None


def convert_length(value: float, src: tx.Any, dst: tx.Any) -> float:
    """
    Convert a length from one unit of space to another.

    Raises
    ------
    ValueError
        If either unit is not a known unit of length.
    """
    a, b = _meters(src), _meters(dst)
    if a is None or b is None:
        raise ValueError(f"Cannot convert a length from {src!r} to {dst!r}.")
    return value * a / b


def _as_unit(unit: tx.Any) -> tx.Optional[Unit]:
    if unit is None or isinstance(unit, Unit):
        return unit
    return Unit(unit)


def _check_size(name: str, size: tx.Any) -> float:
    try:
        size = float(size)
    except (TypeError, ValueError):
        raise ValueError(
            f"The size of axis {name!r} must be a number, not {size!r}."
        ) from None
    if not math.isfinite(size) or size <= 0:
        raise ValueError(
            f"The size of axis {name!r} must be positive and finite, not "
            f"{size!r}."
        )
    return size


def resolve_pixel_size(
    axes: tx.Sequence[Axis],
    metadata: tx.Optional[tx.Mapping[str, AxisScale]] = None,
    pixel_size: tx.Optional[
        tx.Union[float, tx.Sequence[float], tx.Mapping[str, float]]
    ] = None,
    unit: tx.Optional[tx.Union[str, Unit]] = None,
) -> tx.Dict[str, tx.Tuple[float, tx.Optional[Unit]]]:
    """
    Decide the size of a pixel along each spatial axis.

    This applies the caller's keywords over what the file records.

    * `pixel_size` wins over the metadata. It is one size for every
      spatial axis, one per spatial axis (`x, y[, z]`, the brainhops
      order), or a mapping from axis name to size, which overrides only
      the axes it names. Its unit is `unit` if given, or else the unit the
      metadata records for a spatial axis, or else unspecified.
    * Otherwise, the metadata's sizes are used. Given `unit`, they are
      converted to it; if the metadata records no unit, they are taken to
      be in `unit`.
    * `unit` alone, with no size from either, does not make one up: the
      size stays unknown.

    Parameters
    ----------
    axes : Sequence[Axis]
        The axes of the image, in the brainhops order.
    metadata : Mapping[str, AxisScale], optional
        What the file records, by axis name: `(size, unit)`. A reader
        leaves out the axes whose size is unknown, and passes nothing at
        all when it knows none.
    pixel_size : float | Sequence[float] | Mapping[str, float], optional
        The caller's pixel size.
    unit : str | Unit, optional
        The caller's unit of length.

    Returns
    -------
    dict[str, tuple[float, Unit | None]]
        `(size, unit)` by axis name, for the axes whose size is known.
        Empty when no size is known. The metadata may also list
        non-spatial axes, such as a time interval, which are passed
        through unchanged.
    """
    unit = _as_unit(unit)
    space = spatial_axes(axes)
    metadata = {
        name: (_check_size(name, size), _as_unit(u))
        for name, (size, u) in dict(metadata or {}).items()
    }
    out: tx.Dict[str, tx.Tuple[float, tx.Optional[Unit]]] = {}
    space_names = [axis.name for axis in space]

    # The metadata, in `unit` if one is given.
    for name, (size, u) in metadata.items():
        if unit is not None and name in space_names:
            if u is not None:
                size = convert_length(size, u, unit)
            u = unit
        out[name] = (size, u)
    if pixel_size is None:
        return out

    # The caller's sizes, over the metadata.
    if isinstance(pixel_size, tx.Mapping):
        unknown = set(pixel_size) - set(space_names)
        if unknown:
            raise ValueError(
                f"pixel_size names axes that are not spatial axes of "
                f"this image: {sorted(unknown)}. Its spatial axes are "
                f"{space_names}."
            )
        sizes = dict(pixel_size)
    elif isinstance(pixel_size, (Number, np.number)):
        sizes = {name: pixel_size for name in space_names}
    else:
        values = [float(v) for v in np.ravel(pixel_size)]
        if len(values) == 1:
            values = values * len(space_names)
        if len(values) != len(space_names):
            raise ValueError(
                f"pixel_size gives {len(values)} sizes, but the image "
                f"has {len(space_names)} spatial axes {space_names}."
            )
        sizes = dict(zip(space_names, values))
    size_unit = unit
    if size_unit is None:
        known = [
            u
            for name, (_, u) in out.items()
            if name in space_names and u is not None
        ]
        size_unit = known[0] if known else None
    for name, size in sizes.items():
        out[name] = (_check_size(name, size), size_unit)
    return out


def raster_transformations(
    axes: tx.Sequence[Axis],
    scales: tx.Optional[tx.Mapping[str, AxisScale]] = None,
) -> tx.List[Transformation]:
    """
    The voxel-to-world transformations of a raster image.

    The list holds a single
    [`Scaling`][brainhops.datamodel.transformations.Scaling] from the index
    space (`pixel_system`) to the
    physical space
    (`physical_system`). An
    axis listed in `scales` is scaled by its size and measured in its unit;
    any other axis is scaled by one and has no unit. So an image whose size
    is unknown gets the identity, onto a space with no units: one pixel is
    one unit of nothing in particular.

    Parameters
    ----------
    axes : Sequence[Axis]
        The axes of the image, in the brainhops order.
    scales : Mapping[str, AxisScale], optional
        The size and unit of a sample, by axis name (see
        `resolve_pixel_size`).

    Returns
    -------
    list[Transformation]
        The transformations, the preferred one last.
    """
    scales = dict(scales or {})
    pixel = pixel_system(axes)
    physical = physical_system(
        axes, {name: _as_unit(u) for name, (_, u) in scales.items()}
    )
    scale = [float(scales.get(axis.name, (1.0, None))[0]) for axis in axes]
    return [Scaling(input=pixel, output=physical, scale=scale)]


def level_transformation(
    axes: tx.Sequence[Axis],
    scales: tx.Mapping[str, tx.Tuple[float, tx.Any]],
    factors: tx.Sequence[float],
    origin: tx.Mapping[str, float],
) -> Transformation:
    """
    The pixel-to-physical transformation of one level of a pyramid.

    `scales` are the sizes of the full-resolution pixels, by axis name
    (`resolve_pixel_size`), `factors` the downsampling factor of the level
    along each axis, and `origin` the position of the first
    full-resolution pixel, by axis name, in the unit of its size.

    A level downsampled by `f` along an axis has pixels `f` times as large
    as the base level's, and covers the same extent: the edges of the
    first and last pixels coincide. So its pixel `i` is centred on the
    base level's (fractional) pixel `f * i + (f - 1) / 2`, and lands at
    `size * (f * i + (f - 1) / 2) + origin`.
    """
    level_scales = {}
    translation = []
    for axis, factor in zip(axes, factors):
        size, unit = scales.get(axis.name, (1.0, None))
        level_scales[axis.name] = (size * factor, unit)
        translation.append(
            size * (factor - 1) / 2 + origin.get(axis.name, 0.0)
        )
    scaling = raster_transformations(axes, level_scales)[0]
    if not any(translation):
        return scaling
    n = len(axes)
    matrix = np.zeros((n, n + 1))
    matrix[:, :-1] = np.diag(np.asarray(scaling.scale, dtype=float))
    matrix[:, -1] = translation
    return Affine(matrix=matrix, input=scaling.input, output=scaling.output)


def _diagonal(xform: Transformation) -> tx.Optional[np.ndarray]:
    """The per-axis scale of a transformation that is a pure scaling (up
    to a translation), or `None`."""
    if isinstance(xform, Sequence):
        try:
            xform = xform.compute()
        except Exception:
            return None
    if isinstance(xform, Scaling):
        scale = getattr(xform, "scale", None)
        if scale is None:
            return None
        return np.atleast_1d(np.asarray(scale, dtype=float))
    affine = xform
    if not isinstance(affine, Affine):
        try:
            affine = xform.to(Affine)
        except Exception:
            return None
    if not isinstance(affine, Affine):
        return None
    matrix = getattr(affine, "homogeneous_matrix", None)
    if matrix is None:
        return None
    matrix = np.asarray(matrix, dtype=float)[:-1, :-1]
    if matrix.shape[0] != matrix.shape[1]:
        return None
    if np.any(matrix - np.diag(np.diag(matrix))):
        return None
    return np.diag(matrix).copy()


def physical_pixel_size(
    xform: tx.Optional[Transformation],
    axes: tx.Sequence[Axis],
    unit: tx.Union[str, Unit] = "mm",
) -> tx.Optional[tx.Dict[str, float]]:
    """
    The physical size of a pixel along each spatial axis, if it is known.

    It is known when the transformation maps the image's index space by a
    scaling -- possibly with a translation, which a raster format cannot
    store anyway -- onto axes measured in a unit of length. A rotation or
    a shear has no pixel size to give, and an axis with no unit (as an
    image read without one has) has no physical size. A flip is a
    negative scale, whose magnitude is the size.

    Parameters
    ----------
    xform : Transformation, optional
        The image's preferred voxel-to-world transformation.
    axes : Sequence[Axis]
        The axes of the image's data, in the brainhops order.
    unit : str | Unit
        The unit to give the sizes in.

    Returns
    -------
    dict[str, float] | None
        The size along each spatial axis, by name, in `unit`; or `None`
        when it is not known for every spatial axis.
    """
    if xform is None:
        return None
    diag = _diagonal(xform)
    if diag is None or len(diag) != len(axes):
        return None
    output = getattr(getattr(xform, "output", None), "axes", None)
    try:
        output = list(output) if output is not None else None
    except TypeError:
        output = None
    if output is None or len(output) != len(axes):
        return None
    sizes: tx.Dict[str, float] = {}
    for i, axis in enumerate(axes):
        if axis_group(axis) != "space":
            continue
        out_unit = getattr(output[i], "unit", None)
        if _meters(out_unit) is None:
            return None
        size = abs(float(diag[i]))
        if not math.isfinite(size) or size == 0:
            return None
        sizes[axis.name] = convert_length(size, out_unit, unit)
    return sizes or None


# ----------------------------------------------------------------------
#   DOTS PER INCH
# ----------------------------------------------------------------------


def is_default_dpi(dpi: tx.Any) -> bool:
    """
    Whether a resolution is a placeholder that says nothing about the
    pixel size (see `DEFAULT_DPIS`).

    A pair is a placeholder when both of its values are. A missing,
    zero, negative or non-finite resolution is a placeholder too.
    """
    if dpi is None:
        return True
    values = np.ravel(np.asarray(dpi, dtype=float))
    if values.size == 0:
        return True
    for value in values:
        if not math.isfinite(value) or value <= 0:
            return True
    return all(
        any(abs(v - d) <= _DPI_TOLERANCE * d for d in DEFAULT_DPIS)
        for v in values
    )


def dpi_to_size(dpi: float, unit: tx.Union[str, Unit] = "mm") -> float:
    """The size of a pixel, in `unit`, at a resolution in dots per inch."""
    dpi = float(dpi)
    if not math.isfinite(dpi) or dpi <= 0:
        raise ValueError(f"A resolution must be positive, not {dpi!r}.")
    return convert_length(MM_PER_INCH / dpi, "mm", unit)


def size_to_dpi(size: float, unit: tx.Union[str, Unit] = "mm") -> float:
    """The resolution, in dots per inch, of pixels of a size in `unit`."""
    mm = convert_length(float(size), unit, "mm")
    if not math.isfinite(mm) or mm <= 0:
        raise ValueError(f"A pixel size must be positive, not {size!r}.")
    return MM_PER_INCH / mm
