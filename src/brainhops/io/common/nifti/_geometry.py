"""NIfTI geometry: the sform and qform of a voxel-to-world
transformation, units and the image that stores them."""

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel._sugar import get_axes
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import Transformation
from brainhops.datamodel.units import (
    is_indexunit,
    is_physicalunit,
    is_spaceunit,
    is_timeunit,
)
from brainhops.io.base.parsers import UnrepresentableTransformationError
from brainhops.io.common._geometry import (
    AxisLayout,
    arrange_voxel_to_ras,
    declared_axes,
    embed_affine,
)

# this format
from ._constants import (
    _NIFTI_DEFAULT_XFORM_CODE,
    _NIFTI_XFORM_CODE_BY_NAME,
    _QFORM_NAME,
)
from ._header import (
    _apply_like,
    _apply_overrides,
    _new_nifti,
    _NiftiObject,
    _set_other_axes,
)
from ._units import nifti_unit_meters, unit_to_nifti


def _embed_affine(matrix: np.ndarray) -> np.ndarray:
    """
    Embed a homogeneous voxel-to-world matrix in the `(4, 4)` NIfTI stores.

    See [`embed_affine`][brainhops.io.common._geometry.embed_affine].
    """
    return embed_affine(matrix, "NIfTI")


def _voxel_to_ras(
    xform: Transformation, voxel_axes: tx.Optional[tx.List[Axis]] = None
) -> np.ndarray:
    """
    Compute the `(4, 4)` voxel-to-RAS matrix of a transformation.

    See [`_voxel_to_ras_and_others`][], which also returns the scale and
    offset of the axes that follow the spatial ones.
    """
    return _voxel_to_ras_and_others(xform, voxel_axes)[0]


_NIFTI_POLICY = dict(fill_space=True, fill_time=True, time_slot=3)
"""
Where NIfTI stores the axes of an array (see
[`plan_axes`][brainhops.io.common._geometry.plan_axes]).

NIfTI stores the spatial axes first (`dim[1..3]`), then time (`dim[4]`),
then the components of a vector or the channels (`dim[5]`), then any
other axis -- the order the reader declares them in (see `_NIFTI_AXES`).
A slice with other axes is given a `z` axis of size one, so they follow
three spatial axes; and an image with channels, or other axes, but no
time is given a time axis of size one, so they are not read as time --
the `(X, Y, Z, 1, C)` layout of a NIfTI vector image.
"""


def _nifti_geometry(
    xform: Transformation, voxel_axes: tx.Optional[tx.List[Axis]] = None
) -> tx.Tuple[
    np.ndarray,
    tx.List[tx.Optional[tx.Tuple[float, float]]],
    bool,
    tx.Optional[AxisLayout],
]:
    """
    Compute the `(4, 4)` voxel-to-RAS matrix of a transformation, the
    scale and offset of each axis that follows the spatial ones, whether
    the first of those is a time axis, and where each axis of the data is
    stored.

    The transformation must map voxel coordinates to a world space. An
    affine transformation is used directly. A transformation of any other
    kind that reduces to an affine, such as a `Scaling` or a `Sequence` of
    affines, is converted first -- including the sequence of a spatial and
    a temporal subspace transform that a space-and-time image is read as.

    The axes are placed by what the spaces declare, not by position (see
    [`arrange_voxel_to_ras`][brainhops.io.common._geometry.
    arrange_voxel_to_ras]). The voxel axes are `voxel_axes`, the axes of
    the data (see [`_declared_axes`][]), and the world axes are those of
    the transformation's output space. Each side is put in the order NIfTI
    stores (see `_NIFTI_POLICY`): its spatial axes, then its time axis,
    then the channel-like axes, then the others. The caller puts the data
    in the returned layout, which may insert singleton axes: a `z` axis
    for a slice that has other axes, and a time axis for an image with
    channels but no time. A side that declares nothing is taken in NIfTI's
    positional order -- except a world space that declares nothing, of as
    many axes as the voxel space, which is taken to list its axes as the
    voxel space does.

    The NIfTI affine applies to the spatial axes. The axes that follow
    them (time, ...) are split off, each with its scale (its spacing) and
    its offset, which the caller stores apart. A map that mixes the
    spatial axes with the others, or two of the others, or that maps the
    time axis of one space to an axis of another type in the other, has
    no NIfTI form and raises `UnrepresentableTransformationError`. A
    two-dimensional affine is embedded in a `(4, 4)` matrix, which is the
    shape NIfTI stores. The world space is turned into RAS from the
    anatomical orientation of its axes.

    A time axis that the transformation leaves as it is -- scale `1`,
    offset `0` -- and that still counts frames (unit `index`) in the world
    space is not mapped to time: it is returned as `None`, and its
    repetition time is written as missing (see [`_set_other_axes`][]). So
    is a time axis the layout inserted.

    A transformation that has no affine representation, such as a
    displacement field, cannot be written as NIfTI geometry, and raises
    `UnrepresentableTransformationError`.
    """
    arranged = arrange_voxel_to_ras(
        xform, voxel_axes, "NIfTI", **_NIFTI_POLICY
    )
    others = list(arranged.others)

    # >> The axis NIfTI stores fourth is time, unless a space declares
    #    it is not. Both sides must agree on it.
    timed = [
        groups[3] == "time"
        for groups in (arranged.voxel_groups, arranged.world_groups)
        if groups is not None and len(groups) > 3
    ]
    if len(set(timed)) > 1:
        raise UnrepresentableTransformationError(
            "This transformation maps the time axis of one space to an "
            "axis of another type, so it cannot be written as NIfTI "
            "geometry, which maps time to time."
        )
    timed = timed[0] if timed else True

    if (
        timed
        and others
        and others[0] == (1.0, 0.0)
        and (arranged.filled_time or _is_frame_index(arranged.world, 3))
    ):
        # The time axis is not mapped to time: it still counts frames in
        # the world space, so the repetition time is missing.
        others = [None, *others[1:]]
    return arranged.matrix, others, timed, arranged.layout


def _voxel_to_ras_and_others(
    xform: Transformation, voxel_axes: tx.Optional[tx.List[Axis]] = None
) -> tx.Tuple[np.ndarray, tx.List[tx.Optional[tx.Tuple[float, float]]], bool]:
    """
    Compute the `(4, 4)` voxel-to-RAS matrix of a transformation, the
    scale and offset of each axis that follows the spatial ones, and
    whether the first of those is a time axis.

    See [`_nifti_geometry`][], which also returns where each axis of the
    data is stored.
    """
    return _nifti_geometry(xform, voxel_axes)[:3]


def _declared_axes(
    system: tx.Optional[CoordinateSystem], ndim: int
) -> tx.Optional[tx.List[Axis]]:
    """
    The axes of a system, when they say where NIfTI stores each of them.

    See [`declared_axes`][brainhops.io.common._geometry.declared_axes].
    """
    return declared_axes(system, ndim)


def _voxel_axes(
    transformation: Transformation, data: ArrayProtocol
) -> tx.Optional[tx.List[Axis]]:
    """
    The axes of an image's data, when its voxel space declares them.

    They are the axes of the input space of the preferred transformation,
    which indexes the data, when it declares where each goes (see
    [`_declared_axes`][]); `None` otherwise.
    """
    ndim = len(getattr(data, "shape", ()) or ())
    return _declared_axes(getattr(transformation, "input", None), ndim)


def _is_frame_index(system: tx.Optional[CoordinateSystem], k: int) -> bool:
    """Whether axis `k` of a system is a time axis that counts frames."""
    axes = list(getattr(system, "axes", None) or [])
    if k >= len(axes) or axes[k] is Ellipsis:
        return False
    axis = axes[k]
    return getattr(axis, "type", None) == "time" and is_indexunit(
        getattr(axis, "unit", None)
    )


def _reference_code(system: tx.Optional[CoordinateSystem]) -> int:
    """
    The NIfTI xform code for a world space, which is never zero.

    The code names the world reference the matrix maps into, one of
    scanner, aligned, talairach, mni or template. It is taken from the
    world space's name only when that name is one of those references. An
    orientation name, an unnamed space, or an unrecognized reference yields
    the "aligned" code, so a form that carries real geometry is never
    stored with a zero code, which `nibabel` would ignore.
    """
    name = getattr(system, "name", None)
    code = _NIFTI_XFORM_CODE_BY_NAME.get(name, _NIFTI_DEFAULT_XFORM_CODE)
    return code or _NIFTI_DEFAULT_XFORM_CODE


def _sform_and_qform(
    transformations: tx.Sequence[Transformation],
    sform: np.ndarray,
    scode: int,
    voxel_axes: tx.Optional[tx.List[Axis]] = None,
    layout: tx.Optional[AxisLayout] = None,
) -> tx.Tuple[np.ndarray, int, tx.Optional[CoordinateSystem]]:
    """
    Choose the qform matrix, code and world space to store with an sform.

    The matrix and the code always describe the same world space. The
    first transformation other than the preferred one whose world space is
    named after a known reference provides both. Failing that, the rigid
    edge the reader names "qform" provides the matrix, under the sform's
    code. Failing that, the qform is the sform, which `nibabel` reduces to
    its rigid part, under the sform's code.

    The world space that supplies the matrix is returned alongside it, so
    the caller can read the qform's own spatial unit rather than assuming
    it matches the sform's.

    Each matrix is computed with the voxel axes in the order the data is
    written in (`voxel_axes`, see [`_nifti_geometry`][]). A transformation
    that does not place the data's axes as the preferred one does
    (`layout`) -- one that maps another number of axes -- describes
    another array, and is passed over.
    """
    preferred = transformations[-1] if transformations else None

    def _placed(xform: Transformation) -> tx.Optional[np.ndarray]:
        matrix, _, _, placed = _nifti_geometry(xform, voxel_axes)
        if _layout_key(placed) != _layout_key(layout):
            return None
        return matrix

    for xform in transformations:
        if xform is preferred:
            continue
        output = getattr(xform, "output", None)
        name = getattr(output, "name", None)
        if _NIFTI_XFORM_CODE_BY_NAME.get(name):
            matrix = _placed(xform)
            if matrix is not None:
                return matrix, _NIFTI_XFORM_CODE_BY_NAME[name], output

    for xform in transformations:
        output = getattr(xform, "output", None)
        if getattr(output, "name", None) == _QFORM_NAME:
            matrix = _placed(xform)
            if matrix is not None:
                return matrix, scode, output

    return sform, scode, getattr(preferred, "output", None)


def _layout_key(
    layout: tx.Optional[AxisLayout],
) -> tx.Optional[tx.Tuple[tx.Tuple[int, ...], tx.Tuple[int, ...]]]:
    """What a layout does to the data: its order and inserted axes."""
    if layout is None or layout.trivial:
        return None
    return tuple(layout.order), tuple(layout.inserted)


def _space_unit_meters(
    system: tx.Optional[CoordinateSystem],
) -> tx.Optional[float]:
    """
    The size in meters of a world space's first spatial unit, or `None`.

    A space with no physical spatial unit -- every axis unspecified, or
    counting samples -- returns `None`.

    Only the axes the space states can carry a unit, so the `...` of an open
    space reads as it would once closed: as axes with no unit.
    """
    for axis in get_axes(system):
        unit = getattr(axis, "unit", None)
        if is_physicalunit(unit) and is_spaceunit(unit):
            return float(unit.scale)
    return None


def _xyzt_labels(
    system: tx.Optional[CoordinateSystem],
) -> tx.Tuple[str, str]:
    """
    The NIfTI spatial and temporal labels for a world space.

    The first axis with a physical spatial unit gives the spatial label,
    and the first with a physical temporal unit gives the temporal label;
    both go through [`unit_to_nifti`][brainhops.io.common.nifti._units.
    unit_to_nifti], whose policies apply. A spatial unit NIfTI cannot store
    is given the nearest label it can (the affine is rescaled to match, see
    [`_unit_scale`][]). An axis whose unit is unspecified, or counts
    samples, says nothing about either label, which stays `"unknown"`. The
    labels are read from one world space, the preferred transformation's
    output, so a different edge does not change them.
    """
    space = time = None
    # The `...` of an open space carries no unit, as it would once closed.
    for axis in get_axes(system):
        unit = getattr(axis, "unit", None)
        if not is_physicalunit(unit):
            continue
        if space is None and is_spaceunit(unit):
            space = unit_to_nifti(unit, "space", nearest=True)
        elif time is None and is_timeunit(unit):
            time = unit_to_nifti(unit, "time")
    return space or "unknown", time or "unknown"


def _unit_scale(system: tx.Optional[CoordinateSystem], label: str) -> float:
    """
    The factor that rescales a world space's affine into a NIfTI unit.

    A world space measured in one spatial unit is stored under the NIfTI
    label `label`, which is a different unit. The affine is multiplied by
    this factor so the stored geometry keeps the same physical size. A
    space whose unit is unknown, or a label NIfTI cannot store, leaves the
    factor at `1`.
    """
    stored = nifti_unit_meters(label)
    if stored is None:
        return 1.0
    meters = _space_unit_meters(system)
    if meters is None:
        return 1.0
    return meters / stored


def _scale_spatial(matrix: np.ndarray, factor: float) -> np.ndarray:
    """
    Scale the spatial rows of a `(4, 4)` voxel-to-world matrix.

    The three spatial rows carry the world coordinates, so multiplying them
    by the factor converts those coordinates into another spatial unit. The
    bottom row is left unchanged.
    """
    if factor == 1.0:
        return matrix
    scaled = np.array(matrix, dtype=float, copy=True)
    scaled[:3, :] *= factor
    return scaled


def _image_with_geometry(
    data: ArrayProtocol,
    transformation: Transformation,
    transformations: tx.Sequence[Transformation],
    like: tx.Any = None,
    overrides: tx.Optional[tx.Mapping] = None,
) -> _NiftiObject:
    """
    Build a NIfTI image from data and its voxel-to-world geometry.

    The preferred transformation becomes the sform, and its world space
    supplies the sform code. The qform is the rigid edge among the
    transformations when present, and the rigid part of the sform
    otherwise.

    The NIfTI affine applies to the spatial axes. The spacing of each
    axis that follows them, and the origin of the time axis (`toffset`),
    are read from the preferred transformation too.

    The axes are written in the order NIfTI stores them: the spatial axes,
    then time, then the channels, then the others. When the voxel space of
    the preferred transformation declares its axes in another order, such
    as `(t, x, y, z)`, the data is transposed into that order (lazily, for
    a lazy array), and every form is written for the transposed data. A
    slice with other axes is given a `z` axis of size one, and an image
    with channels but no time a time axis of size one, so `(x, y, t)` is
    written `(X, Y, 1, T)`, `(x, y, z, c)` `(X, Y, Z, 1, C)` and `(x, y,
    c)` `(X, Y, 1, 1, C)` (see [`_nifti_geometry`][]). A voxel space that
    declares nothing is written in the order it has, as NIfTI's positional
    convention reads it (see [`_declared_axes`][]).

    The spatial and temporal units are read from the preferred
    transformation's output space. A spatial unit NIfTI cannot store is
    converted to the nearest one it can, and each form's affine is scaled
    from its own world space's unit, so the stored geometry keeps its
    physical size.

    Non-encoding header fields are taken from `like` when it is given, and
    caller `overrides` are applied last so an explicit value wins.
    """
    preferred_output = getattr(transformation, "output", None)
    space, time = _xyzt_labels(preferred_output)

    voxel_axes = _voxel_axes(transformation, data)
    sform_raw, others, timed, layout = _nifti_geometry(
        transformation, voxel_axes
    )
    scode = _reference_code(preferred_output)
    qform_raw, qcode, qform_output = _sform_and_qform(
        transformations, sform_raw, scode, voxel_axes, layout
    )
    if layout is not None:
        data = layout.apply(data)

    sform = _scale_spatial(sform_raw, _unit_scale(preferred_output, space))
    qform = _scale_spatial(qform_raw, _unit_scale(qform_output, space))

    image = _new_nifti(data, sform)
    _set_other_axes(image, others, timed)
    _apply_like(image, like)
    image.header.set_sform(sform, code=scode)
    image.header.set_qform(qform, code=qcode)
    image.header.set_xyzt_units(space, time)
    _apply_overrides(image, overrides)
    return image
