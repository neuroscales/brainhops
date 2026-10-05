"""The element type a writer stores an array as."""

__all__ = ["preferred_dtype"]

# externals
import numpy as np
import typing_extensions as tx

# internals
from ._filebased import FileBasedMetadata
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
    changed = (
        metadata._changed_fields()
        if isinstance(metadata, FileBasedMetadata)
        else {"data_type": wanted}
    )
    if "data_type" in changed:
        report = ConversionReport(target=getattr(metadata, "format", None))
        report.approximated["data_type"] = (
            f"stored as {array_dtype.name}: the values are not "
            f"{wanted.name} values"
        )
        apply_loss_policy(report, on_loss, stacklevel=2)
    return array_dtype


def _same_kind(source: np.dtype, target: np.dtype) -> bool:
    """Whether values of `source` may be stored as `target` without
    changing their kind: integers (booleans included) as integers,
    floats as floats, complex numbers as complex numbers."""

    def kind(dtype: np.dtype) -> str:
        return "i" if dtype.kind in "biu" else dtype.kind

    return kind(source) == kind(target) and target.kind != "b"
