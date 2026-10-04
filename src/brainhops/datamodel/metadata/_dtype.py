"""The element type a writer stores an array as."""

__all__ = ["preferred_dtype"]

# externals
import numpy as np
import typing_extensions as tx

# internals
from ._filebased import FileBasedMetadata
from ._report import ConversionReport
from ._sentinel import UNSUPPORTED


def preferred_dtype(
    metadata: tx.Any,
    array_dtype: tx.Any,
    dtype: tx.Any = None,
    *,
    report: tx.Optional[ConversionReport] = None,
) -> np.dtype:
    """
    The element type a writer stores an array as.

    In order: an explicit `dtype` (the writer option); the `data_type` of
    the metadata (the type the file had when it was read, or the one
    set), when the array's values are of its kind, so that a label map
    read as `uint8` is written as `uint8` again but a resampled, floating
    point version of it is not quantised; the array's own type. A
    `data_type` set (or converted) by hand that is not used is reported
    in `report` as approximated; one that was only read is dropped
    silently (the data changed kind since the read).
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
    if report is not None:
        changed = (
            metadata.changed_fields()
            if isinstance(metadata, FileBasedMetadata)
            else {"data_type": wanted}
        )
        if "data_type" in changed:
            report.approximated["data_type"] = (
                f"stored as {array_dtype.name}: the values are not "
                f"{wanted.name} values"
            )
    return array_dtype


def _same_kind(source: np.dtype, target: np.dtype) -> bool:
    """Whether values of `source` may be stored as `target` without
    changing their kind: integers (booleans included) as integers,
    floats as floats, complex numbers as complex numbers."""

    def kind(dtype: np.dtype) -> str:
        return "i" if dtype.kind in "biu" else dtype.kind

    return kind(source) == kind(target) and target.kind != "b"
