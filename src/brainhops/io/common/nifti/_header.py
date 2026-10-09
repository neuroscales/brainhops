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

# typing
_NiftiObject = tx.Union[nb.Nifti1Header, nb.Nifti1Image]


def _nifti_intent(header: "_NiftiObject") -> tx.Optional[int]:
    """The intent code of a NIfTI header, or `None` if unreadable."""
    try:
        if isinstance(header, nb.Nifti1Image):
            header = header.header
        return int(header["intent_code"])
    except Exception:
        return None


def _nifti_intent_name(header: "_NiftiObject") -> tx.Optional[str]:
    """The intent name of a NIfTI header, or `None` if unreadable."""
    try:
        if isinstance(header, nb.Nifti1Image):
            header = header.header
        return str(header.get_intent()[2])
    except Exception:
        return None


def _nifti_vector_field(data: ArrayProtocol) -> ArrayProtocol:
    """
    Drop the singleton time axis of a NIfTI vector field.

    NIfTI stores a 3-D vector field as `(X, Y, Z, 1, 3)`, with the
    components in the fifth axis and a singleton in place of time. Every
    reader of a field wants it as `(X, Y, Z, 3)`, or it would sample it
    as a 4-D grid of 3-vectors, so the singleton is dropped here, in one
    place. Any other array -- a 4-D `(X, Y, Z, 3)` field, a 5-D one with
    several time points -- is returned as it is, for the caller to
    accept or refuse.

    The shared `NiftiParser.data` keeps the axis on purpose: it is the
    array that matches the header, and is what writers that copy a file
    back store.
    """
    shape = tuple(int(d) for d in data.shape)
    if len(shape) == 5 and shape[3] == 1:
        data = data[:, :, :, 0, :]
    return data


def _nifti_shape(header: "_NiftiObject") -> tx.Optional[tx.Tuple[int, ...]]:
    """The data shape of a NIfTI header, or `None` if unreadable."""
    try:
        if isinstance(header, nb.Nifti1Image):
            header = header.header
        return tuple(int(d) for d in header.get_data_shape())
    except Exception:
        return None


def _nifti_to_axes(header: nb.Nifti1Header) -> tx.List[Axis]:
    """
    Compute the axes of a NIfTI file, based on its header.

    This function takes into account the intent code of the NIfTI file
    to change the names and types of the axes, if necessary.

    Axes that are deemed irrelevant by the intent code are given a name
    of `None`. Every axis is an axis of the voxel space, so its unit is
    the index unit.
    """

    ndim = len(header.get_data_shape())

    # Get names and types of existing axes
    axes = _NIFTI_AXES[:ndim]

    # Apply specific knowledge from intent code.
    # `header[...]` hands back a 0-d numpy array, which is unhashable and
    # cannot be looked up in a dict, so go through `_nifti_intent`.
    intent = _nifti_intent(header)
    if intent in _NIFTI_SPECIFIC_AXES:
        axes_map = _NIFTI_SPECIFIC_AXES[intent]
        axes = [axes_map.get(i, axis) for i, axis in enumerate(axes)]

    # The tables above are module-level templates: hand out copies, so a
    # system built from them never holds -- and never changes -- the
    # shared ones.
    return [replace(axis) for axis in axes]


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


def _new_nifti(
    data: ArrayProtocol, affine: tx.Optional[np.ndarray]
) -> _NiftiObject:
    """
    Build a `nibabel` image, reporting an unwritable dtype as a writer error.

    The array is stored as it is given. Any object that follows the array
    protocol is accepted, including a `cupy` or `dask` array, so a lazy or
    device array is not materialized into `numpy` here.

    NIfTI-1 records each dimension in a signed 16-bit field. An array whose
    extent along any axis exceeds that limit is written as NIfTI-2, which
    stores dimensions in 64-bit fields. A smaller array is written as
    NIfTI-1.

    NIfTI-1 cannot store some array types, such as 64-bit integers.
    `nibabel` raises a bare `ValueError` for one, which is re-raised as a
    `WriterError` that names the dtype.
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
    """
    Store the scale and offset of the axes that follow the spatial ones.

    The scale of each becomes its spacing (`pixdim`), and the offset of
    the first, when it is the time axis (`timed`), becomes `toffset`.
    NIfTI stores no origin for any other axis, so a nonzero offset there,
    or a negative spacing, raises `UnrepresentableTransformationError`. A
    transformation over more axes than the data has raises `WriterError`.

    A time axis that is not mapped to time (`None` in `others`, see
    [`_voxel_to_ras_and_others`][]) has no repetition time: its spacing is
    written as `0`, which NIfTI reads as missing, and `toffset` is left
    alone.
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
    """
    Copy non-encoding header fields from a template onto an image.

    The geometry of the written image always comes from the object being
    written, so the sform, the qform and their codes are never copied. The
    description is taken from the template. The intent is taken from the
    template only when the image has none of its own, so a field's own
    intent is preserved.

    Fields that change how the voxels are read back are never copied. The
    data scaling (`scl_slope`, `scl_inter`) rescales every value, and the
    stored data type reinterprets the bytes, so both are left as the image
    computed them from its own array.

    The image is returned so calls can be chained.
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
    """
    Apply caller-supplied header overrides onto an image.

    The overrides are applied last, after the derived geometry and after
    any `like` template, so an explicit value always wins. `dtype` sets the
    stored data type, `intent` sets the intent code, and `descrip` sets the
    description. Any other name is written straight to the header field of
    that name.

    The array's own data type is kept unless `dtype` is given, so the
    override is opt in. The image is returned so calls can be chained.
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
