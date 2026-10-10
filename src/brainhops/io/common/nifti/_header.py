"""NIfTI headers: intents, shapes, axes, and building a header
or image from another."""

# dependencies
import nibabel as nb
import numpy as np
import typing_extensions as tx
from bagof.magic import replace
from nibabel.arrayproxy import ArrayProxy

# internals
from brainhops._core import path
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel.axes import Axis
from brainhops.io.base.parsers import (
    UnrepresentableTransformationError,
    WriterError,
)

# this format
from ._constants import (
    _NIFTI1_MAX_DIM,
    _NIFTI_AXES,
    _NIFTI_INTENT_NAME_MAPPING,
    _NIFTI_INTENT_VECTOR,
    _NIFTI_SPECIFIC_AXES,
)
from ._raw import NiftiRaw

_NiftiObject = tx.Union[nb.Nifti1Header, nb.Nifti1Image]


def _nifti_intent(header: "_NiftiObject") -> tx.Optional[int]:
    """Return the intent code of a header, or `None` if unreadable."""
    try:
        if isinstance(header, nb.Nifti1Image):
            header = header.header
        return int(header["intent_code"])
    except Exception:
        return None


def _nifti_intent_name(header: "_NiftiObject") -> tx.Optional[str]:
    """Return the intent name of a header, or `None` if unreadable."""
    try:
        if isinstance(header, nb.Nifti1Image):
            header = header.header
        return str(header.get_intent()[2])
    except Exception:
        return None


def _nifti_shape(header: "_NiftiObject") -> tx.Optional[tx.Tuple[int, ...]]:
    """Return the data shape of a header, or `None` if unreadable."""
    try:
        if isinstance(header, nb.Nifti1Image):
            header = header.header
        return tuple(int(d) for d in header.get_data_shape())
    except Exception:
        return None


def _nifti_to_axes(header: nb.Nifti1Header) -> tx.List[Axis]:
    """Return the voxel axes described by a header.

    The intent code can change the names and types of the axes. Every axis
    counts samples, so its unit is the index unit.
    """

    ndim = len(header.get_data_shape())

    axes = _NIFTI_AXES[:ndim]

    # Indexing the header gives an unhashable 0-d array, so the intent is read
    # with `_nifti_intent`.
    intent = _nifti_intent(header)
    if intent in _NIFTI_SPECIFIC_AXES:
        axes_map = _NIFTI_SPECIFIC_AXES[intent]
        axes = [axes_map.get(i, axis) for i, axis in enumerate(axes)]

    # Copy the shared templates so that no system can mutate them.
    return [replace(axis) for axis in axes]


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


def _new_nifti(
    data: ArrayProtocol,
    affine: tx.Optional[np.ndarray],
    header: tx.Optional[nb.Nifti1Header] = None,
) -> _NiftiObject:
    """Build a nibabel image, as NIfTI-2 if an extent exceeds NIfTI-1.

    The array is stored as given, so lazy and device arrays are not
    materialised. nibabel chooses the stored data type from the array and
    refuses the types that NIfTI cannot store. When a header is given, the
    image is then built again over that header, which receives the chosen
    data type. A NIfTI-2 header gives a NIfTI-2 image whatever the extents.

    Raises
    ------
    WriterError
        If NIfTI cannot store the data type of the array.
    """
    shape = tuple(int(d) for d in getattr(data, "shape", ()) or ())
    image_cls = nb.Nifti1Image
    if isinstance(header, nb.Nifti2Header) or any(
        d > _NIFTI1_MAX_DIM for d in shape
    ):
        image_cls = nb.Nifti2Image
    try:
        image = image_cls(data, affine)
    except ValueError as error:
        dtype = getattr(data, "dtype", "unknown")
        raise WriterError(
            f"NIfTI cannot store an array of type {dtype}: {error}"
        ) from error
    if header is None:
        return image
    header.set_data_dtype(image.get_data_dtype())
    return image_cls(data, affine, header)


def _set_other_axes(
    image: _NiftiObject,
    others: tx.Sequence[tx.Optional[tx.Tuple[float, float]]],
    timed: bool = True,
) -> None:
    """Store the scale and offset of the axes after the spatial ones.

    Scales become spacings, and the offset of a time axis becomes `toffset`. A
    time axis that is not mapped to time gets a spacing of zero, which NIfTI
    reads as missing.

    Raises
    ------
    UnrepresentableTransformationError
        If a spacing is negative or an axis other than time has an offset.
    WriterError
        If the transformation maps more axes than the data have.
    """
    if not others:
        return
    header = image.header
    zooms = list(header.get_zooms())
    if 3 + len(others) > len(zooms):
        raise WriterError(
            f"The voxel-to-world transformation maps {3 + len(others)} "
            f"axes, but the data has {len(zooms)}."
        )
    for k, other in enumerate(others):
        if other is None:
            zooms[3 + k] = 0.0
            continue
        scale, offset = other
        if scale < 0:
            raise UnrepresentableTransformationError(
                f"NIfTI stores the spacing of axis {3 + k} as a positive "
                f"number, so a map that reverses it ({scale}) cannot be "
                f"written."
            )
        if (k or not timed) and offset != 0:
            raise UnrepresentableTransformationError(
                f"NIfTI stores an origin for the time axis only, so a map "
                f"that shifts axis {3 + k} ({offset}) cannot be written."
            )
        zooms[3 + k] = scale
    header.set_zooms(zooms)
    if timed and others[0] is not None:
        header["toffset"] = others[0][1]


def _header(obj: tx.Any) -> tx.Any:
    """Return the header of an object, or `None` when it has none.

    The NIfTI formats, such as a `NiftiImage` or a `NiftiVoxelToRAS`, hold
    their header in the record of their metadata. A nibabel image and the
    formats that are still built on a parser, such as an MGH image, have a
    `header` attribute instead.
    """
    record = getattr(getattr(obj, "metadata", None), "raw", None)
    header = getattr(record, "header", None)
    if header is not None:
        return header
    return getattr(obj, "header", None)


def _like_header(like: tx.Any) -> tx.Optional[nb.Nifti1Header]:
    """Return the header to copy from a `like` template, or `None`.

    The template is a nibabel header or image, an object with a header, an
    object whose metadata holds a NIfTI record, such as a `NiftiImage`, or
    the path of a NIfTI file, whose header alone is read.
    """
    if like is None:
        return None
    if isinstance(like, (nb.Nifti1Header, nb.Nifti2Header)):
        return like
    header = _header(like)
    if header is not None:
        return header
    if isinstance(like, (str, path.PathLike)):
        return NiftiRaw.from_filename(like).header
    return None


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


def _record_image(
    raw: ArrayProtocol,
    record: tx.Optional[NiftiRaw],
    affine: tx.Optional[np.ndarray] = None,
    overrides: tx.Optional[tx.Mapping[str, tx.Any]] = None,
) -> _NiftiObject:
    """Build the nibabel image that writes an array over a record.

    The header of the record is the base of the new header, so that its
    description, its intent and its extensions are kept. That header is
    changed, so the caller passes a copy of the record, such as the one
    that `NiftiMetadata.to_raw` returns. When the geometry is given as
    the voxel-to-world `affine`, the fields that describe the layout of
    the array and the geometry are reset and encoded again, and the
    caller then sets the codes and the intent that its format writes.
    Without `affine`, the geometry was not changed, so the sform and the
    qform of the record are kept with their codes, and only the shape
    and the data type follow the array. Without a record, the header is
    new, and its affine is `affine` or the identity.

    A nibabel proxy is written unscaled, with the data type and the scaling
    that the file stored, so that an untouched file is written byte for
    byte. When the `overrides`, which the caller applies afterwards, change
    the data type or the scaling, the proxy is read and scaled again
    instead.
    """
    header = None if record is None else record.header
    scaling = None
    if isinstance(raw, ArrayProxy) and not (
        _STORAGE_OVERRIDES.intersection(overrides or {})
    ):
        scaling = _stored_scaling(header, raw)
        raw = raw.get_unscaled()
    if header is None:
        affine = np.eye(4) if affine is None else affine
        image = _new_nifti(raw, affine)
    elif affine is None:
        # The scaling of the record describes the array that the file
        # stored. Another array is scaled again by nibabel.
        header.set_slope_inter(None, None)
        image = _new_nifti(raw, None, header)
    else:
        _reset_structural_fields(header)
        image = _new_nifti(raw, affine, header)
    if scaling is not None:
        image.header["scl_slope"], image.header["scl_inter"] = scaling
    return image


def _set_coordinates_intent(header: nb.Nifti1Header) -> None:
    """Give a header the intent of a field of coordinates.

    The intent is `VECTOR` (1007) with the intent name `"Mapping"`. A
    header that already has this intent is left as it is, so that its
    intent parameters are kept.
    """
    intent = (_nifti_intent(header), _nifti_intent_name(header))
    if intent != (_NIFTI_INTENT_VECTOR, _NIFTI_INTENT_NAME_MAPPING):
        header.set_intent(
            _NIFTI_INTENT_VECTOR, name=_NIFTI_INTENT_NAME_MAPPING
        )


def _apply_like(image: _NiftiObject, like: tx.Any) -> _NiftiObject:
    """Copy the description and intent of a template onto an image.

    The intent is copied only if the image has none. Fields that change the
    geometry or how the voxels are read back are never copied. The image is
    returned for chaining.
    """
    header = _like_header(like)
    if header is None:
        return image
    target = image.header
    try:
        target["descrip"] = header["descrip"]
    except (KeyError, ValueError):
        pass
    if int(target["intent_code"]) == 0:
        for field in (
            "intent_code",
            "intent_name",
            "intent_p1",
            "intent_p2",
            "intent_p3",
        ):
            try:
                target[field] = header[field]
            except (KeyError, ValueError):
                pass
    return image


def _apply_overrides(
    image: _NiftiObject, overrides: tx.Mapping
) -> _NiftiObject:
    """Apply header overrides and return the image.

    The keys `dtype`, `intent` and `descrip` set the stored data type, the
    intent and the description; any other key sets the header field of that
    name.
    """
    overrides = dict(overrides or {})
    dtype = overrides.pop("dtype", None)
    intent = overrides.pop("intent", None)
    descrip = overrides.pop("descrip", None)
    if dtype is not None:
        image.header.set_data_dtype(dtype)
    if intent is not None:
        image.header.set_intent(intent)
    if descrip is not None:
        if isinstance(descrip, str):
            descrip = descrip.encode("utf-8", "replace")
        image.header["descrip"] = descrip
    for field, value in overrides.items():
        image.header[field] = value
    return image
