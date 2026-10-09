"""NIfTI geometry: the sform and qform of a voxel-to-world
transformation, units and the image that stores them."""

# dependencies
import nibabel as nb
import numpy as np
import typing_extensions as tx
from nibabel.arrayproxy import ArrayProxy

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
from ._raw import NiftiRaw
from ._units import nifti_unit_meters, unit_to_nifti


def _embed_affine(matrix: np.ndarray) -> np.ndarray:
    """Embed a voxel-to-world matrix in the `(4, 4)` matrix that NIfTI stores,
    with [`embed_affine`][].
    """
    return embed_affine(matrix, "NIfTI")


def _voxel_to_ras(
    xform: Transformation, voxel_axes: tx.Optional[tx.List[Axis]] = None
) -> np.ndarray:
    """Return the `(4, 4)` voxel-to-RAS matrix of a transformation."""
    return _voxel_to_ras_and_others(xform, voxel_axes)[0]


_NIFTI_POLICY = dict(fill_space=True, fill_time=True, time_slot=3)
"""Where NIfTI stores the array axes.

The policy is passed to [`arrange_voxel_to_ras`][]. The spatial axes come
first, then time, then the vector components or channels, then any other axis.
A slice with other axes gets a `z` of size one, and an image with channels but
no time gets a time axis of size one.
"""


def _nifti_geometry(
    xform: Transformation, voxel_axes: tx.Optional[tx.List[Axis]] = None
) -> tx.Tuple[
    np.ndarray,
    tx.List[tx.Optional[tx.Tuple[float, float]]],
    bool,
    tx.Optional[AxisLayout],
]:
    """Compute the NIfTI geometry of a voxel-to-world transformation.

    The transformation is reduced to an affine map, and the axes of each side
    are placed by what the spaces declare, in the order of [`_NIFTI_POLICY`][].
    The affine part applies to the spatial axes, and each later axis gets a
    scale and an offset. A time axis that the transformation leaves unchanged
    and that still counts frames is not mapped to time, so its repetition time
    is written as missing.

    Returns
    -------
    matrix : numpy.ndarray
        The `(4, 4)` voxel-to-RAS matrix.
    others : list
        The scale and offset of each later axis, or `None`.
    timed : bool
        Whether the first later axis is time.
    layout : AxisLayout or None
        The layout in which the data must be stored.

    Raises
    ------
    UnrepresentableTransformationError
        If the transformation has no NIfTI form, for example because it maps
        time to another type of axis.
    """
    arranged = arrange_voxel_to_ras(
        xform, voxel_axes, "NIfTI", **_NIFTI_POLICY
    )
    others = list(arranged.others)

    # The fourth axis is time unless a space says otherwise, and both sides
    # must agree.
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
        others = [None, *others[1:]]
    return arranged.matrix, others, timed, arranged.layout


def _voxel_to_ras_and_others(
    xform: Transformation, voxel_axes: tx.Optional[tx.List[Axis]] = None
) -> tx.Tuple[np.ndarray, tx.List[tx.Optional[tx.Tuple[float, float]]], bool]:
    """Return the first three results of [`_nifti_geometry`][]."""
    return _nifti_geometry(xform, voxel_axes)[:3]


def _declared_axes(
    system: tx.Optional[CoordinateSystem], ndim: int
) -> tx.Optional[tx.List[Axis]]:
    """Return the axes of a system if they say where NIfTI stores each."""
    return declared_axes(system, ndim)


def _voxel_axes(
    transformation: Transformation, data: ArrayProtocol
) -> tx.Optional[tx.List[Axis]]:
    """Return the declared axes of the input space of a transformation, which
    indexes the data, or `None`.
    """
    ndim = len(getattr(data, "shape", ()) or ())
    return _declared_axes(getattr(transformation, "input", None), ndim)


def _is_frame_index(system: tx.Optional[CoordinateSystem], k: int) -> bool:
    """Return whether axis `k` of a system is a time axis counting frames."""
    axes = list(getattr(system, "axes", None) or [])
    if k >= len(axes) or axes[k] is Ellipsis:
        return False
    axis = axes[k]
    return getattr(axis, "type", None) == "time" and is_indexunit(
        getattr(axis, "unit", None)
    )


def _reference_code(system: tx.Optional[CoordinateSystem]) -> int:
    """Return the non-zero xform code of a world space.

    A space named after a known reference gets its code; any other space is
    stored as `aligned`.
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
    """Choose the qform matrix, code and world space for an sform.

    The first transformation, other than the preferred last one, whose world
    space is named after a known reference is chosen. Otherwise the rigid
    `qform` transformation is used under the sform code, and failing that the
    sform itself. A transformation that places the data axes differently from
    `layout` is skipped. The world space is returned so that the caller can
    read its unit.
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
    """Return the order and inserted axes of a layout, or `None` if trivial."""
    if layout is None or layout.trivial:
        return None
    return tuple(layout.order), tuple(layout.inserted)


def _space_unit_meters(
    system: tx.Optional[CoordinateSystem],
) -> tx.Optional[float]:
    """Return the size in metres of the first spatial unit of a world space, or
    `None` if it has no physical spatial unit.
    """
    for axis in get_axes(system):
        unit = getattr(axis, "unit", None)
        if is_physicalunit(unit) and is_spaceunit(unit):
            return float(unit.scale)
    return None


def _xyzt_labels(
    system: tx.Optional[CoordinateSystem],
) -> tx.Tuple[str, str]:
    """Return the NIfTI spatial and temporal unit labels of a space.

    The labels come from the first axes with a physical spatial and temporal
    unit, through [`unit_to_nifti`][]. A spatial unit that NIfTI cannot store
    gets the nearest label, and a missing unit gives `"unknown"`.
    """
    space = time = None
    # The `...` of an open space carries no unit.
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
    """Return the factor that converts a world space to the unit of a NIfTI
    label, or 1 if either unit is unknown.
    """
    stored = nifti_unit_meters(label)
    if stored is None:
        return 1.0
    meters = _space_unit_meters(system)
    if meters is None:
        return 1.0
    return meters / stored


def _scale_spatial(matrix: np.ndarray, factor: float) -> np.ndarray:
    """Scale the spatial rows of a `(4, 4)` matrix by `factor`."""
    if factor == 1.0:
        return matrix
    scaled = np.array(matrix, dtype=float, copy=True)
    scaled[:3, :] *= factor
    return scaled


# Fields of a header that describe how the voxels are laid out and stored,
# and the geometry that the model encodes. A record used as the base of a
# header has these fields reset, and the writer sets them again. The other
# fields, such as the description, the intent, the calibration range, the
# slice timing, `toffset` and the extensions, are kept from the record.
_STRUCTURAL_FIELDS = (
    "dim",
    "pixdim",
    "datatype",
    "bitpix",
    "vox_offset",
    "scl_slope",
    "scl_inter",
    "magic",
    "xyzt_units",
    "qform_code",
    "sform_code",
    "quatern_b",
    "quatern_c",
    "quatern_d",
    "qoffset_x",
    "qoffset_y",
    "qoffset_z",
    "srow_x",
    "srow_y",
    "srow_z",
)


# Overrides that change how the voxels are stored, so that a proxy cannot be
# copied as it is stored.
_STORAGE_OVERRIDES = frozenset({"dtype", "scl_slope", "scl_inter"})


def _reset_structural_fields(header: nb.Nifti1Header) -> None:
    """Give the structural fields of a header their values in a new header.

    The fields are listed in [`_STRUCTURAL_FIELDS`][]. A header reset in
    this way keeps everything else that its file recorded, and the writer
    then encodes the layout of the data and the geometry over it.
    """
    fresh = type(header)()
    for name in _STRUCTURAL_FIELDS:
        header[name] = fresh[name]


def _stored_scaling(
    header: tx.Optional[nb.Nifti1Header], proxy: ArrayProxy
) -> tx.Tuple[tx.Any, tx.Any]:
    """Return the `scl_slope` and `scl_inter` to write with unscaled voxels.

    The voxels of a proxy are written as they are stored, so the file needs
    the slope and the intercept that the proxy applies when it reads them.
    The values of the record are kept when they mean the same scaling, so
    that a slope of zero, which means that the voxels are not scaled, is
    written as zero again rather than as one.
    """
    if header is not None:
        slope, inter = header.get_slope_inter()
        slope = 1.0 if slope is None else slope
        inter = 0.0 if inter is None else inter
        if (slope, inter) == (proxy.slope, proxy.inter):
            # A field of a header is a view, which a reset would change.
            return header["scl_slope"].item(), header["scl_inter"].item()
    return proxy.slope, proxy.inter


def _image_with_geometry(
    data: ArrayProtocol,
    transformation: Transformation,
    transformations: tx.Sequence[Transformation],
    base: tx.Optional[NiftiRaw] = None,
    like: tx.Any = None,
    overrides: tx.Optional[tx.Mapping] = None,
) -> _NiftiObject:
    """Build a NIfTI image from data and their voxel-to-world geometry.

    The header is built in a fixed order. The record `base`, when it is
    given, is the starting point: its structural fields are reset with
    [`_reset_structural_fields`][], and its other fields and extensions are
    kept. The record is changed in place, so the caller passes a copy.
    Without a record, the header starts empty. The geometry is then encoded
    over the header, header fields are copied from `like`, and the
    `overrides` are applied last.

    The preferred transformation provides the sform, the spacing of the
    later axes and the units, and the qform comes from
    [`_sform_and_qform`][]. A spatial unit that NIfTI cannot store is
    converted, and each matrix is scaled accordingly. A record whose
    spatial unit is unknown keeps it unknown when the model is in
    millimetres, because the reader reads an unknown spatial unit as
    millimetres.

    The axes are written in NIfTI order, so data whose voxel space declares
    another order are transposed (lazily for lazy arrays). Singleton axes are
    inserted where NIfTI needs them: `(x,y,t)->(X,Y,1,T)`,
    `(x,y,z,c)->(X,Y,Z,1,C)` and `(x,y,c)->(X,Y,1,1,C)`.

    The data is written after the header. A nibabel proxy that reaches the
    writer unchanged is written unscaled, with the data type, the slope and
    the intercept that the file stored, so that an untouched image is
    written byte for byte. An override of the data type or of the scaling
    makes the proxy be read and scaled again instead.
    """
    overrides = dict(overrides or {})
    header = None if base is None else base.header

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

    scaling = None
    if isinstance(data, ArrayProxy) and not (
        _STORAGE_OVERRIDES.intersection(overrides)
    ):
        scaling = _stored_scaling(header, data)
        data = data.get_unscaled()
    if header is not None:
        if space == "mm" and header.get_xyzt_units()[0] == "unknown":
            space = "unknown"
        _reset_structural_fields(header)

    image = _new_nifti(data, sform, header)
    _set_other_axes(image, others, timed)
    _apply_like(image, like)
    image.header.set_sform(sform, code=scode)
    image.header.set_qform(qform, code=qcode)
    image.header.set_xyzt_units(space, time)
    _apply_overrides(image, overrides)
    if scaling is not None:
        image.header["scl_slope"], image.header["scl_inter"] = scaling
    return image
