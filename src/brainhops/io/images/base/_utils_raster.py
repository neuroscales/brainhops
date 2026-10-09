"""Conventions shared by the raster image formats.

Raster formats such as PNG, JPEG, TIFF or WebP store an array of samples
and at most a pixel size, without an origin, an orientation or a world
space. This module defines how every raster reader interprets such files,
as [`brainhops.io.common.nifti`][brainhops.io.common.nifti] does for NIfTI,
so that a backend only has to decode the file to a C-ordered array and a
description of its axes.

1. An integer index is the centre of a pixel, as in NIfTI and ITK, so
   pixel `(0, 0)` covers `[-0.5, 0.5]` along both axes.
2. Readers return a view of the decoded `(y, x)` array with its axes in
   the brainhops order, which [`to_canonical`][] computes. In that order,
   the spatial axes come first, fastest first, and are followed by time,
   channels and other axes. Writers transpose the data back with
   [`to_storage`][].
3. Row zero is the top row, so `y` points down. No orientation is
   attached to the axes.
4. The components of a pixel lie along a
   [`ChannelAxis`][brainhops.datamodel.axes.ChannelAxis] named `"c"`,
   which a single-component image does not have.
5. The voxel-to-world transformation is one
   [`Scaling`][brainhops.datamodel.transformations.Scaling] onto a system
   named `"physical"`, which carries the pixel size and unit when the
   format records a meaningful size.
6. When the size is absent or a placeholder (72, 96 or 1 dpi), the scaling
   is the identity and the physical axes have no unit.
7. The `pixel_size` and `unit` options of every reader override the file
   metadata ([`resolve_pixel_size`][]).

A reader proceeds as follows, with storage axes given as codes spelled as
in tifffile or as a list of [`Axis`][brainhops.datamodel.axes.Axis]:

```python
axes = storage_axes("ZYXS")                    # C order
data, axes = to_canonical(array, axes)         # F-ordered view: x, y, z, c
scales = resolve_pixel_size(axes, metadata, pixel_size=..., unit=...)
transformations = raster_transformations(axes, scales)
```

A writer uses [`image_axes`][], [`to_storage`][] and
[`physical_pixel_size`][] in the converse direction.
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

import math
from numbers import Number

import numpy as np
import typing_extensions as tx
from bagof.magic import replace

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
"""Millimeters per inch, which relate a resolution to a pixel size."""

DEFAULT_DPIS: tx.Tuple[float, ...] = (1.0, 72.0, 96.0)
"""Resolutions that carry no physical meaning.

Writers without size information use 72 dpi (the historical default of
screens), 96 dpi (Windows) or 1 dpi (tifffile). A resolution equal to one
of these is read as unknown.
"""

_DPI_TOLERANCE = 1e-3
# PNG stores an integer number of pixels per meter, so 72 dpi comes back
# as 72.009 (2835 px/m). A resolution relatively close to a placeholder
# therefore counts as that placeholder.

_INDEX = "index"

AxisScale = tx.Tuple[float, tx.Optional[tx.Union[str, Unit]]]
"""The size of one sample along an axis, and its unit.

A unit of `None` means that the unit is unspecified, in which case the size
has no physical meaning.
"""

_AxesLike = tx.Union[str, tx.Sequence[Axis]]

# Single-letter storage codes, spelled as in tifffile, mapped to an axis
# name and type. Unlisted codes give an axis of unknown type.
_CODE_ROLES: tx.Dict[str, tx.Tuple[str, tx.Optional[str]]] = {
    "X": ("x", "space"),
    "Y": ("y", "space"),
    "Z": ("z", "space"),
    "T": ("t", "time"),
    "C": ("c", "channel"),
    "S": ("c", "channel"),  # samples per pixel, as in RGB
}

# Canonical order of the axis groups, as in the NIfTI and Zarr readers.
_GROUPS = ("space", "time", "channel", "other")

_SPACE_RANK = {"x": 0, "y": 1, "z": 2}

# Axis types whose components are colours or vector components.
_CHANNEL_TYPES = ("channel", "displacement", "coordinate")


# ----------------------------------------------------------------------
#   AXES
# ----------------------------------------------------------------------


def storage_axes(codes: _AxesLike) -> tx.List[Axis]:
    """Build the axes of a C-ordered array from its storage codes.

    Each character describes one axis, slowest first, spelled as in
    tifffile. `X`, `Y` and `Z` are spatial and `T` is time. `C` and `S`
    (samples per pixel) give a channel axis `"c"`, except that `S` becomes
    `"s"` when `C` is also present. Other letters give axes of unknown type,
    named in lower case, and repeated names are numbered (`"q"`, `"q1"`).
    A sequence of [`Axis`][] is returned unchanged as a list.

    Raises
    ------
    TypeError
        If a sequence contains an element that is not an [`Axis`][].
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
    """Return the group of an axis: space, time, channel or other.

    Displacement and coordinate axes belong to the channel group.
    """
    type_ = getattr(axis, "type", None)
    if type_ in _CHANNEL_TYPES:
        return "channel"
    if type_ in ("space", "time"):
        return type_
    return "other"


def spatial_axes(axes: tx.Sequence[Axis]) -> tx.List[Axis]:
    """Return the spatial axes among `axes`, in their order."""
    return [axis for axis in axes if axis_group(axis) == "space"]


def canonical_permutation(axes: _AxesLike) -> tx.List[int]:
    """Return the permutation from storage axes to the brainhops order.

    Spatial axes come first, then time, channels and other axes. Within a
    group, axes are sorted fastest first, except that named spatial axes
    always sort as `x`, `y`, `z`. A `"YXS"` array thus becomes `x`, `y`,
    `c`. Element `i` of the result is the storage position of the axis
    that becomes axis `i`, for both the array and the list of axes.
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
    """Transpose a C-ordered array to the brainhops order without a copy.

    The storage axes are listed slowest first, as for
    [`storage_axes`][]. A `"YX"` image or a `"ZYX"` volume comes back
    exactly in Fortran order.

    Returns
    -------
    data : array
        Transposed view.
    axes : list of Axis
        Axes of `data`.

    Raises
    ------
    ValueError
        If the number of axes differs from the number of dimensions.
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
    """Return the permutation from brainhops axes to a storage order.

    This function is the converse of [`canonical_permutation`][], for
    writers with a fixed storage order such as `"YXS"`. Each storage axis
    takes the image axis with the same name. Failing that, a storage axis
    named `x`, `y` or `z` takes the first, second or third spatial axis,
    and any other takes the first remaining axis of its group. Element `i`
    of the result is the position in `axes` of the axis stored at `i`.

    Raises
    ------
    ValueError
        If the numbers of axes differ, or if a storage axis has no match.
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
    for j, target in enumerate(targets):
        name = getattr(target, "name", None)
        if name is not None and name in names and names.index(name) in left:
            perm[j] = names.index(name)
            left.remove(perm[j])
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
    """Transpose data from the brainhops order to a storage order.

    The result is a view, which may need
    [`numpy.ascontiguousarray`][numpy.ascontiguousarray].
    """
    return array.transpose(storage_permutation(axes, codes))


def default_axes(ndim: int, channel: bool = False) -> tx.List[Axis]:
    """Return the axes assumed for an array that is not described.

    The axes are named `x`, `y`, `z`, `t` and `c`, in that order, for as
    many dimensions as there are, and any further axis `i` is named
    `dim<i>`. With `channel=True`, the last axis is a channel axis, and
    the other axes follow the same defaults.
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
    """Return the axes of `system` if it has one per dimension, else `None`.

    The system is usually the input of the preferred transformation. When
    the result is `None`, the writer applies its own default (see
    [`default_axes`][]).
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
    """Return the coordinate system of the raster index space.

    The axes count samples, fastest first. The system is named `"pixel"`
    when there are two spatial axes, and `"voxel"` otherwise.
    """
    axes = [replace(axis, unit=_INDEX) for axis in axes]
    name = "pixel" if len(spatial_axes(axes)) == 2 else "voxel"
    return CoordinateSystem(name=name, axes=axes, order="F")


def physical_system(
    axes: tx.Sequence[Axis],
    units: tx.Optional[tx.Mapping[str, tx.Any]] = None,
) -> CoordinateSystem:
    """Return the `"physical"` system, onto which pixel sizes map indices.

    Each axis takes the unit that `units` gives for its name, or no unit.
    """
    units = dict(units or {})
    out = [replace(axis, unit=units.get(axis.name)) for axis in axes]
    return CoordinateSystem(name="physical", axes=out)


# ----------------------------------------------------------------------
#   PIXEL SIZE
# ----------------------------------------------------------------------


def _meters(unit: tx.Any) -> tx.Optional[float]:
    """Return the meters per unit of a length unit, or `None`."""
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
    """Convert a length between two space units.

    Raises
    ------
    ValueError
        If either unit is not a known length unit.
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
    """Decide the pixel size of each spatial axis.

    The caller's `pixel_size` takes precedence over the file metadata. It
    is one size for every spatial axis, one size per spatial axis, or a
    mapping that overrides the named axes only. Its unit is `unit`, or
    else the unit that the metadata records for a spatial axis. The sizes
    that the metadata records for spatial axes are converted to `unit`
    when it is given, and they are taken to be in `unit` if they have no
    unit. A `unit` given without any size never makes up a size.

    Returns
    -------
    dict
        Mapping from axis name to `(size, unit)` for the axes of known
        size. Metadata of other axes, such as a time interval, is passed
        through.

    Raises
    ------
    ValueError
        If a size is not positive and finite, or if `pixel_size` does not
        match the spatial axes.
    """
    unit = _as_unit(unit)
    space = spatial_axes(axes)
    metadata = {
        name: (_check_size(name, size), _as_unit(u))
        for name, (size, u) in dict(metadata or {}).items()
    }
    out: tx.Dict[str, tx.Tuple[float, tx.Optional[Unit]]] = {}
    space_names = [axis.name for axis in space]

    for name, (size, u) in metadata.items():
        if unit is not None and name in space_names:
            if u is not None:
                size = convert_length(size, u, unit)
            u = unit
        out[name] = (size, u)
    if pixel_size is None:
        return out

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
    """Return the voxel-to-world transformations of a raster image.

    The list holds one [`Scaling`][] from [`pixel_system`][] to
    [`physical_system`][]. Axes listed in `scales` are scaled by their
    size and take its unit; the other axes are scaled by one, without a
    unit.
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
    """Return the pixel-to-physical transformation of a pyramid level.

    A level downsampled by `f` has pixels `f` times larger and the same
    extent, so its pixel `i` is centred on base pixel `f * i + (f - 1) / 2`.
    `scales` holds the full-resolution sizes, and `origin` holds the
    position of the first full-resolution pixel, both by axis name. The
    result is a [`Scaling`][], or an [`Affine`][] when there is a
    translation.
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
    """Return the scale of each axis of a pure scaling, or `None`."""
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
    """Return the physical pixel size of each spatial axis, if known.

    The size is known when `xform` is a scaling, possibly with a
    translation, onto axes in a length unit. The magnitude of a negative
    scale (a flip) is the size. The result is `None` for a rotation or
    shear, or when any spatial axis has no length unit.
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
    """Return whether a resolution is a placeholder that gives no size.

    A resolution is a placeholder when every value matches one of
    [`DEFAULT_DPIS`][], or when it is missing or has a value that is not
    positive and finite.
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
    """Return the pixel size, in `unit`, at a resolution in dpi."""
    dpi = float(dpi)
    if not math.isfinite(dpi) or dpi <= 0:
        raise ValueError(f"A resolution must be positive, not {dpi!r}.")
    return convert_length(MM_PER_INCH / dpi, "mm", unit)


def size_to_dpi(size: float, unit: tx.Union[str, Unit] = "mm") -> float:
    """Return the resolution in dpi of pixels of a size given in `unit`."""
    mm = convert_length(float(size), unit, "mm")
    if not math.isfinite(mm) or mm <= 0:
        raise ValueError(f"A pixel size must be positive, not {size!r}.")
    return MM_PER_INCH / mm
