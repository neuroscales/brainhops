"""The element type, and the intensity scaling, that a writer stores an
array with."""

__all__ = ["preferred_storage", "stored_values"]

# externals
import numpy as np
import typing_extensions as tx

# internals
from brainhops.datamodel.metadata import ConversionReport, Metadata


def preferred_storage(
    metadata: tx.Optional[Metadata],
    data: tx.Any,
    *,
    report: tx.Optional[ConversionReport] = None,
) -> tx.Tuple[np.dtype, tx.Optional[float], tx.Optional[float]]:
    """
    Choose how a writer stores an array: its element type and its scaling.

    The element type is `metadata.dtype`, and the type of the data when
    the metadata gives none. An override given to `save` or to
    `consolidate` is a field of the metadata by the time a writer calls
    this function, so it takes precedence over the type that the file
    was read with. A label map read as `uint8` and computed on as
    `int64` is therefore written as `uint8` again.

    The metadata describes the scaling of the file with `scale_slope`
    and `scale_intercept`. The scaling is kept when the element type is
    an integer type and every value of the array is, up to a thousandth
    of a step, `stored * scale_slope + scale_intercept` for an integer
    `stored` that the type can hold, so a scaled integer file that was
    read and left unchanged is written back exactly as it was. Otherwise
    the array is stored without a scaling, and the scaling is reported
    as approximated.

    Parameters
    ----------
    metadata : Metadata, optional
        The metadata of the image being written.
    data : array-like
        The values to write. They are read only when a scaling has to be
        checked.
    report : ConversionReport, optional
        The report of the write, which receives the scaling that could
        not be kept.

    Returns
    -------
    dtype : numpy.dtype
        The element type to store the array as.
    slope : float or None
        The slope of the scaling, or `None` for none.
    intercept : float or None
        The intercept of the scaling, or `None` for none. With a scaling,
        the writer stores `(value - intercept) / slope` as `dtype` (see
        [`stored_values`][]).
    """
    dtype = getattr(metadata, "dtype", None)
    if dtype is None:
        dtype = getattr(data, "dtype", np.float64)
    dtype = np.dtype(dtype)
    slope = getattr(metadata, "scale_slope", None)
    intercept = getattr(metadata, "scale_intercept", None)
    if slope in (None, 1.0) and intercept in (None, 0.0):
        return dtype, None, None
    slope = 1.0 if slope is None else float(slope)
    intercept = 0.0 if intercept is None else float(intercept)
    if _fits(data, dtype, slope, intercept):
        return dtype, slope, intercept
    if report is not None:
        for name in ("scale_slope", "scale_intercept"):
            if getattr(metadata, name, None) is not None:
                report.approximated[name] = (
                    f"stored unscaled as {dtype.name}: the values do not "
                    f"fit the scaling"
                )
    return dtype, None, None


def stored_values(
    data: tx.Any,
    dtype: tx.Any,
    slope: tx.Optional[float] = None,
    intercept: tx.Optional[float] = None,
) -> np.ndarray:
    """
    Return the values that a writer stores, with a type and a scaling.

    Each value is stored as `(value - intercept) / slope`, rounded to the
    nearest integer only when `dtype` is an integer type, then cast to
    `dtype`.

    Parameters
    ----------
    data : array-like
        The values to write.
    dtype : dtype-like
        The element type to store them as.
    slope : float, optional
        The slope of the scaling, or `None` for a slope of 1.
    intercept : float, optional
        The intercept of the scaling, or `None` for an intercept of 0.

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


def _fits(
    data: tx.Any, dtype: np.dtype, slope: float, intercept: float
) -> bool:
    """Return whether every value is an integer of `dtype` once unscaled."""
    if dtype.kind not in "iu" or not slope:
        return False
    try:
        values = np.asarray(data, dtype=np.float64)
    except Exception:
        return False
    if not values.size:
        return True
    if not np.all(np.isfinite(values)):
        return False
    stored = (values - intercept) / slope
    rounded = np.round(stored)
    info = np.iinfo(dtype)
    return bool(
        np.all(np.abs(stored - rounded) <= 1e-3)
        and rounded.min() >= info.min
        and rounded.max() <= info.max
    )
