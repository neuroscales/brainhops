"""NIfTI headers: intents, shapes, axes, and building a header
or image from another."""

# dependencies
import nibabel as nb
import numpy as np
import typing_extensions as tx
from bagof.magic import replace

# internals
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
    _NIFTI_SPECIFIC_AXES,
)
from ._files import _like_header

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


def _nifti_vector_field(data: ArrayProtocol) -> ArrayProtocol:
    """Drop the singleton time axis of a five-dimensional array.

    NIfTI stores a vector field with shape `(X, Y, Z, 1, 3)`, whereas readers
    expect `(X, Y, Z, 3)`. Any other array is returned unchanged. Unlike this
    function, [`NiftiParser.data`][] keeps the axis so that it matches the
    header.
    """
    shape = tuple(int(d) for d in data.shape)
    if len(shape) == 5 and shape[3] == 1:
        data = data[:, :, :, 0, :]
    return data


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
    data: ArrayProtocol, affine: tx.Optional[np.ndarray]
) -> _NiftiObject:
    """Build a nibabel image, as NIfTI-2 if an extent exceeds NIfTI-1.

    The array is stored as given, so lazy and device arrays are not
    materialised.

    Raises
    ------
    WriterError
        If NIfTI cannot store the data type of the array.
    """
    shape = tuple(int(d) for d in getattr(data, "shape", ()) or ())
    image_cls = nb.Nifti1Image
    if any(d > _NIFTI1_MAX_DIM for d in shape):
        image_cls = nb.Nifti2Image
    try:
        return image_cls(data, affine)
    except ValueError as error:
        dtype = getattr(data, "dtype", "unknown")
        raise WriterError(
            f"NIfTI cannot store an array of type {dtype}: {error}"
        ) from error


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
