"""The element type, and the intensity scaling, a writer stores an
array with."""

__all__ = ["preferred_dtype", "preferred_storage", "stored_values"]

# externals
import numpy as np
import typing_extensions as tx

# internals
from ._report import ConversionReport, OnLoss, apply_loss_policy
from ._sentinel import UNSUPPORTED


def preferred_dtype(
    metadata: tx.Any,
    array_dtype: tx.Any,
    dtype: tx.Any = None,
    *,
    on_loss: tx.Optional[OnLoss] = None,
) -> np.dtype:
    """
    Choose the element type that a writer stores an array as.

    An explicit `dtype`, the writer option, always wins. Otherwise, the
    `data_type` of the metadata is used when the values of the array are
    of the same kind (integers, floating-point numbers or complex
    numbers). The `data_type` is the type the file had when it was read,
    or the type set since, so a label map read as `uint8` is written as
    `uint8` again, while a resampled floating-point version of the same
    map is not quantised. In every other case the array keeps its own
    type.

    A `data_type` that was set or converted by hand, but cannot be used
    because the values are of another kind, is reported as approximated.
    A `data_type` that was only read is dropped silently, since the data
    changed kind after the read.

    Parameters
    ----------
    metadata : Metadata or None
        The metadata of the object being written.
    array_dtype : dtype-like
        The element type of the array.
    dtype : dtype-like, optional
        The element type requested by the caller of the writer.
    on_loss : {"ignore", "warn", "raise"} or ConversionReport, optional
        What to do with an unused `data_type` that was set by hand. By
        default, the policy in effect; a writer passes the report of its
        write.

    Returns
    -------
    numpy.dtype
        The element type to store the array as.

    Raises
    ------
    MetadataLossError
        If an unused `data_type` is reported under the `"raise"` policy.
    """
    array_dtype = np.dtype(array_dtype)
    if dtype is not None:
        return np.dtype(dtype)
    wanted = getattr(metadata, "data_type", None)
    if wanted is None or wanted is UNSUPPORTED:
        return array_dtype
    wanted = np.dtype(wanted)
    if _same_kind(array_dtype, wanted):
        return wanted
    changed = _changed_fields(metadata, {"data_type": wanted})
    if "data_type" in changed:
        report = ConversionReport(target=getattr(metadata, "format", None))
        report.approximated["data_type"] = (
            f"stored as {array_dtype.name}: the values are not "
            f"{wanted.name} values"
        )
        apply_loss_policy(report, on_loss, stacklevel=2)
    return array_dtype


def preferred_storage(
    metadata: tx.Any,
    data: tx.Any,
    dtype: tx.Any = None,
    *,
    on_loss: tx.Optional[OnLoss] = None,
) -> tx.Tuple[np.dtype, tx.Optional[float], tx.Optional[float]]:
    """
    Choose how a writer that supports an intensity scaling stores an
    array: its element type, and the scaling.

    The metadata describes the storage of the file that was read with three
    fields: `data_type`, `scale_slope` and `scale_intercept`. Without a
    scaling, the element type is chosen as [`preferred_dtype`][] chooses
    it. With a scaling, the array is stored as `data_type` with that
    scaling when `data_type` is an integer type and every value of the
    array is, up to a thousandth of a step, `stored * scale_slope +
    scale_intercept` for an integer `stored` that `data_type` can hold. A
    scaled integer file that was read and left unchanged is therefore
    written back exactly as it was. When the values do not fit the scaling
    (after a resampling, for instance), the array is stored unscaled, and
    the storage fields that were changed by hand are reported as
    approximated.

    Parameters
    ----------
    metadata : Metadata or None
        The metadata of the image being written.
    data : array-like
        The values to write. They are read only when a scaling has to be
        checked.
    dtype : dtype-like, optional
        The element type requested by the caller of the writer. It wins
        over the metadata, and no scaling is applied.
    on_loss : {"ignore", "warn", "raise"} or ConversionReport, optional
        What to do with storage fields that were set by hand but cannot
        be used. By default, the policy in effect; a writer passes the
        report of its write.

    Returns
    -------
    dtype : numpy.dtype
        The element type to store the array as.
    slope, intercept : float or None
        The scaling to store the array with, or `None` for none. When a
        scaling is returned, the writer stores `(value - intercept) /
        slope` as `dtype`, rounded only when `dtype` is an integer type
        (see [`stored_values`][]). A scaling is only returned with an
        integer type, the one case where it makes sense.

    Raises
    ------
    MetadataLossError
        If an unused storage field is reported under the `"raise"`
        policy.
    """
    array_dtype = np.dtype(getattr(data, "dtype", np.float64))
    if dtype is not None:
        return np.dtype(dtype), None, None
    slope = _present(getattr(metadata, "scale_slope", None))
    intercept = _present(getattr(metadata, "scale_intercept", None))
    if slope in (None, 1.0) and intercept in (None, 0.0):
        return (
            preferred_dtype(metadata, array_dtype, on_loss=on_loss),
            None,
            None,
        )
    slope = 1.0 if slope is None else float(slope)
    intercept = 0.0 if intercept is None else float(intercept)
    wanted = _present(getattr(metadata, "data_type", None))
    if wanted is not None and _fits(data, np.dtype(wanted), slope, intercept):
        return np.dtype(wanted), slope, intercept
    report = ConversionReport(target=getattr(metadata, "format", None))
    stored = preferred_dtype(metadata, array_dtype, on_loss=report)
    changed = _changed_fields(
        metadata, {"scale_slope": slope, "scale_intercept": intercept}
    )
    for name in ("scale_slope", "scale_intercept"):
        if name in changed and changed[name] is not None:
            report.approximated[name] = (
                f"stored unscaled as {stored.name}: the values do not fit "
                f"the scaling"
            )
    apply_loss_policy(report, on_loss, stacklevel=2)
    return stored, None, None


def stored_values(
    data: tx.Any,
    dtype: tx.Any,
    slope: tx.Optional[float] = None,
    intercept: tx.Optional[float] = None,
) -> np.ndarray:
    """
    The values a writer stores, with the type and the scaling that
    [`preferred_storage`][] chose.

    Each value is stored as `(value - intercept) / slope`, rounded to the
    nearest integer only when `dtype` is an integer type, then cast to
    `dtype`.

    Parameters
    ----------
    data : array-like
        The values to write.
    dtype : dtype-like
        The element type to store them as.
    slope, intercept : float, optional
        The scaling, or `None` for none (a slope of 1 and an intercept of
        0).

    Returns
    -------
    numpy.ndarray
        The stored values, of type `dtype`.
    """
    dtype = np.dtype(dtype)
    if slope is None and intercept is None:
        return np.asarray(data).astype(dtype, copy=False)
    slope = 1.0 if slope is None else float(slope)
    intercept = 0.0 if intercept is None else float(intercept)
    stored = (np.asarray(data, dtype=np.float64) - intercept) / slope
    if dtype.kind in "iu":
        stored = np.round(stored)
    return stored.astype(dtype)


# ----------------------------------------------------------------------
#   PRIVATE
# ----------------------------------------------------------------------


def _same_kind(source: np.dtype, target: np.dtype) -> bool:
    """Whether values of `source` may be stored as `target` without
    changing their kind: integers (booleans included) as integers,
    floats as floats, complex numbers as complex numbers."""

    def kind(dtype: np.dtype) -> str:
        return "i" if dtype.kind in "biu" else dtype.kind

    return kind(source) == kind(target) and target.kind != "b"


def _present(value: tx.Any) -> tx.Any:
    return None if value is UNSUPPORTED else value


def _fits(
    data: tx.Any, dtype: np.dtype, slope: float, intercept: float
) -> bool:
    """Whether every value is an integer of `dtype` once unscaled."""
    if dtype.kind not in "iu" or not slope:
        return False
    try:
        values = np.asarray(data, dtype=np.float64)
    except Exception:
        return False
    if not values.size or not np.all(np.isfinite(values)):
        return bool(values.size == 0)
    stored = (values - intercept) / slope
    rounded = np.round(stored)
    info = np.iinfo(dtype)
    return bool(
        np.all(np.abs(stored - rounded) <= 1e-3)
        and rounded.min() >= info.min
        and rounded.max() <= info.max
    )


def _changed_fields(
    metadata: tx.Any, assigned: tx.Dict[str, tx.Any]
) -> tx.Dict[str, tx.Any]:
    """
    The fields of `metadata` that changed since it was read.

    The metadata of a file format (`FileBasedMetadata`, in `brainhops.io`,
    which this package does not import) knows what changed since the
    read, with `_changed_fields`. For any other metadata, every field in
    `assigned` counts as changed.
    """
    changed_fields = getattr(metadata, "_changed_fields", None)
    if changed_fields is None:
        return assigned
    return changed_fields()
