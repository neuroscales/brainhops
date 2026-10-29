"""The common vocabulary in the header of a NIfTI file.

[`NiftiMetadata`][brainhops.io.common.nifti.NiftiMetadata] decodes the
fields of the vocabulary from the header of its record with [`decode`][]
when it is built, and encodes the fields that changed into a copy of the
header with [`encode`][] when it is consolidated. The slots are these.

| Field                      | Header slots                          |
|----------------------------|---------------------------------------|
| `description`              | `descrip` (80 bytes)                  |
| `sources`                  | `aux_file` (one name of 24 bytes)     |
| `intent`                   | `intent_code`                         |
| `space`                    | `sform_code`, or else `qform_code`    |
| `display_range`            | `cal_min`, `cal_max`                  |
| `phase_encoding_direction` | `dim_info` (an axis, no polarity)     |
| `slice_encoding_direction` | `dim_info` (an axis, no polarity)     |
| `slice_timing`             | `slice_code` and the other `slice_*`  |
| `repetition_time`          | `pixdim[4]`, in the unit of the time  |
| `dtype`                    | `datatype`, `bitpix`                  |
| `scale_slope`              | `scl_slope`                           |
| `scale_intercept`          | `scl_inter`                           |

Some slots belong to the geometry or to the axes, which the image or the
transformation owns, so the metadata does not write them. The world
space names the codes of the sform and of the qform, so a changed
`space` is reported as approximated. An intent that changes how the axes
are read (a displacement field or a vector, for instance) is reported in
the same way. A field that NIfTI has no slot for, `extra` among them, is
left to the caller, which reports it as lost.
"""

__all__ = ["clear_spatial_slots", "decode", "encode"]

# dependencies
import nibabel as nb
import numpy as np
import typing_extensions as tx

# internals
from brainhops._core.numeric import float32_repr
from brainhops.datamodel.metadata import ConversionReport, EncodingDirection

# this format
from ._constants import NIFTI_FIELD_INTENTS, NIFTI_SPECIFIC_AXES
from ._raw import NiftiRaw


def decode(raw: tx.Optional[NiftiRaw]) -> tx.Dict[str, tx.Any]:
    """Return the fields of the vocabulary that a NIfTI header holds.

    Parameters
    ----------
    raw : NiftiRaw, optional
        The record. Without a record, no field is decoded.

    Returns
    -------
    dict
        Field name to value, for the fields that the header sets.
    """
    if raw is None:
        return {}
    h = raw.header
    out: tx.Dict[str, tx.Any] = {}
    out["description"] = _text(h, "descrip")
    aux = _text(h, "aux_file")
    out["sources"] = (aux,) if aux else None
    code = int(h["intent_code"])
    out["intent"] = _intent_label(code) if code else None
    out["space"] = _space(h)
    cal = (float32_repr(h["cal_min"]), float32_repr(h["cal_max"]))
    out["display_range"] = cal if any(cal) else None
    out["dtype"] = h.get_data_dtype()
    out["scale_slope"], out["scale_intercept"] = _scaling(h)
    _, phase, slice_ = h.get_dim_info()
    if phase is not None and phase < len(_AXES):
        out["phase_encoding_direction"] = _AXES[phase]
    if slice_ is not None and slice_ < len(_AXES):
        out["slice_encoding_direction"] = _AXES[slice_]
        out["slice_timing"] = _slice_timing(h)
    scale = _time_scale(h)
    if len(_shape(h)) >= 4 and scale is not None:
        step = float32_repr(h["pixdim"][4])
        if step > 0:
            out["repetition_time"] = step * scale
    return out


def encode(
    raw: NiftiRaw, changed: tx.Dict[str, tx.Any], report: ConversionReport
) -> NiftiRaw:
    """Write the changed fields into the header of a record.

    Each field that the header takes is written, and its name is taken
    out of `changed`. A field whose value has no slot is left in
    `changed`, so that the caller reports it as lost, and the slot it
    would have replaced is cleared. A field that the header stores only
    approximately is reported in `report`.

    Parameters
    ----------
    raw : NiftiRaw
        The record to write into, which is already a copy.
    changed : dict
        Field name to new value, for the fields that differ from the
        header. `None` clears a slot.
    report : ConversionReport
        The report of the consolidation.

    Returns
    -------
    NiftiRaw
        The record.
    """
    h = raw.header
    if "description" in changed:
        _set_text(h, "descrip", changed.pop("description"), 80, report)
    if "sources" in changed:
        sources = changed.pop("sources")
        if sources and len(sources) > 1:
            report.approximated["sources"] = (
                f"only the first of {len(sources)} kept (aux_file)"
            )
        first = sources[0] if sources else None
        _set_text(h, "aux_file", first, 24, report, name="sources")
    if "display_range" in changed:
        low, high = changed.pop("display_range") or (0.0, 0.0)
        h["cal_min"], h["cal_max"] = low, high
    for name, position in (
        ("phase_encoding_direction", 1),
        ("slice_encoding_direction", 2),
    ):
        if name in changed:
            _set_dim_info(h, name, changed, position, report)
    if "slice_timing" in changed:
        _set_slice_timing(h, changed, report)
    if "repetition_time" in changed:
        _set_repetition_time(h, changed)
    if "intent" in changed:
        _set_intent(h, changed.pop("intent"), report)
    if "space" in changed:
        report.approximated["space"] = (
            f"the codes of the sform and the qform name {_space(h)!r}, "
            f"and the geometry decides them"
        )
        changed.pop("space")
    if "dtype" in changed:
        _set_dtype(h, changed, report)
    if "scale_slope" in changed or "scale_intercept" in changed:
        slope, intercept = _scaling(h)
        slope = changed.pop("scale_slope", slope)
        intercept = changed.pop("scale_intercept", intercept)
        h["scl_slope"] = np.nan if slope is None else slope
        h["scl_inter"] = np.nan if slope is None else (intercept or 0.0)
    return raw


def clear_spatial_slots(raw: NiftiRaw) -> NiftiRaw:
    """Return a copy of a record without the slots tied to the voxel axes.

    The slots are `dim_info` and the slice slots, which the metadata of an
    image keeps until the spatial axes of the image move.

    Parameters
    ----------
    raw : NiftiRaw
        The record, which is not changed.

    Returns
    -------
    NiftiRaw
        The copy.
    """
    raw = raw.copy()
    raw.header.set_dim_info(None, None, None)
    _clear_slices(raw.header)
    return raw


# ----------------------------------------------------------------------
#   PRIVATE
# ----------------------------------------------------------------------

_AXES = "ijk"

# The NIfTI xform codes that name a world space. Code 0 means that the
# form is unused.
_XCODES = {
    1: "scanner",
    2: "aligned",
    3: "talairach",
    4: "mni",
    5: "template",
}

# Intent codes that change how the axes are read; the geometry decides
# them, never the metadata.
_AXIS_INTENTS = frozenset(NIFTI_SPECIFIC_AXES) | NIFTI_FIELD_INTENTS

# Seconds per NIfTI time unit.
_TIME_UNITS = {"sec": 1.0, "msec": 1e-3, "usec": 1e-6}


def _text(header: nb.Nifti1Header, slot: str) -> tx.Optional[str]:
    value = np.asarray(header[slot]).item()
    if isinstance(value, bytes):
        value = value.split(b"\0", 1)[0].decode("utf-8", "replace")
    value = str(value).strip()
    return value or None


def _space(header: nb.Nifti1Header) -> tx.Optional[str]:
    scode, qcode = int(header["sform_code"]), int(header["qform_code"])
    return _XCODES.get(scode) or _XCODES.get(qcode)


def _scaling(
    header: nb.Nifti1Header,
) -> tx.Tuple[tx.Optional[float], tx.Optional[float]]:
    """Return the slope and the intercept of the scaling of a header.

    A slope of 0 or NaN means no scaling, as the NIfTI standard says, and
    so does the identity.
    """
    slope = float(header["scl_slope"])
    intercept = float(header["scl_inter"])
    if not slope or not np.isfinite(slope):
        return None, None
    if not np.isfinite(intercept):
        intercept = 0.0
    if slope == 1.0 and intercept == 0.0:
        return None, None
    return float32_repr(slope), float32_repr(intercept) or None


def _time_scale(header: nb.Nifti1Header) -> tx.Optional[float]:
    """Return the seconds per unit of the time axis, or `None`."""
    try:
        return _TIME_UNITS.get(header.get_xyzt_units()[1])
    except Exception:
        return None


def _intent_label(code: int) -> tx.Optional[str]:
    try:
        return str(nb.nifti1.intent_codes.label[code])
    except KeyError:
        return None


def _intent_code(name: str) -> tx.Optional[int]:
    try:
        return int(nb.nifti1.intent_codes.code[name])
    except KeyError:
        return None


def _shape(header: nb.Nifti1Header) -> tx.Tuple[int, ...]:
    return tuple(int(d) for d in header.get_data_shape())


def _slice_timing(
    header: nb.Nifti1Header,
) -> tx.Optional[tx.Tuple[float, ...]]:
    scale = _time_scale(header) or 1.0
    try:
        if not int(header["slice_code"]) or not float(
            header["slice_duration"]
        ):
            return None
        times = header.get_slice_times()
    except Exception:
        return None
    if any(t is None for t in times):
        # Padding slices have no time: BIDS has no way to say so, and
        # the record keeps them.
        return None
    # The times are multiples of the slice duration, which is stored in
    # single precision: they are computed again in double precision from
    # the duration as written.
    stored = float(header["slice_duration"])
    duration = float32_repr(stored) * scale
    return tuple(round(round(float(t) / stored) * duration, 9) for t in times)


def _set_text(
    header: nb.Nifti1Header,
    slot: str,
    value: tx.Optional[str],
    size: int,
    report: ConversionReport,
    name: str = "description",
) -> None:
    data = (value or "").encode("utf-8")
    if len(data) > size:
        report.approximated[name] = f"truncated to {size} bytes ({slot})"
        data = data[:size]
    header[slot] = data


def _clear_slices(header: nb.Nifti1Header) -> None:
    for slot in ("slice_code", "slice_start", "slice_end", "slice_duration"):
        header[slot] = 0


def _set_dim_info(
    header: nb.Nifti1Header,
    name: str,
    changed: tx.Dict[str, tx.Any],
    position: int,
    report: ConversionReport,
) -> None:
    """Write an encoding direction as the voxel axis of `dim_info`.

    A direction along no voxel axis has no slot: the slot is cleared, and
    the direction is left in `changed`.
    """
    value: tx.Optional[EncodingDirection] = changed[name]
    dims = list(header.get_dim_info())
    bids = None if value is None else value.to_bids()
    if value is not None and bids is None:
        dims[position] = None
        header.set_dim_info(*dims)
        return
    changed.pop(name)
    if bids is None:
        dims[position] = None
    else:
        if bids.endswith("-"):
            report.approximated[name] = "polarity dropped (dim_info)"
        dims[position] = _AXES.index(bids[0])
    header.set_dim_info(*dims)


def _set_slice_timing(
    header: nb.Nifti1Header,
    changed: tx.Dict[str, tx.Any],
    report: ConversionReport,
) -> None:
    """Write a slice timing as the slice slots.

    A timing without a slice axis in `dim_info`, without one time per
    slice, or in an order that NIfTI cannot name, has no slot: the slots
    are cleared, and the timing is left in `changed`.
    """
    value = changed["slice_timing"]
    if value is None:
        _clear_slices(header)
        changed.pop("slice_timing")
        return
    scale = _time_scale(header) or 1.0
    times = [float(t) / scale for t in value]
    try:
        if header.get_dim_info()[2] is None:
            raise ValueError("no slice axis (dim_info)")
        header.set_slice_times(times)
    except Exception:
        _clear_slices(header)
        return
    changed.pop("slice_timing")


def _set_repetition_time(
    header: nb.Nifti1Header, changed: tx.Dict[str, tx.Any]
) -> None:
    """Write a repetition time as the time step, `pixdim[4]`.

    A header with fewer than four dimensions has no slot, and the time is
    left in `changed`. `None` writes a step of zero, which NIfTI reads as
    a missing repetition time. A header without a time unit is given
    seconds.
    """
    value = changed["repetition_time"]
    if len(_shape(header)) < 4:
        return
    changed.pop("repetition_time")
    if value is None:
        header["pixdim"][4] = 0.0
        return
    space, time = header.get_xyzt_units()
    if time not in _TIME_UNITS:
        header.set_xyzt_units(space, "sec")
        time = "sec"
    header["pixdim"][4] = float(value) / _TIME_UNITS[time]


def _set_intent(
    header: nb.Nifti1Header,
    value: tx.Optional[str],
    report: ConversionReport,
) -> None:
    """Write an intent, unless the intent belongs to the axes.

    An intent that changes how the axes are read is decided by the
    geometry, so neither such an intent nor a change of an intent of
    the record that is one of them is written; both are reported as
    approximated. An intent that NIfTI does not name is reported as
    lost.
    """
    current = int(header["intent_code"])
    if current in _AXIS_INTENTS:
        report.approximated["intent"] = (
            f"the axes keep the intent {_intent_label(current)!r}"
        )
        return
    if value is None:
        header.set_intent(0)
        return
    code = _intent_code(value)
    if code is None:
        report.lost["intent"] = value
    elif code in _AXIS_INTENTS:
        report.approximated["intent"] = (
            f"{value!r} changes how the axes are read, which the geometry "
            f"decides"
        )
    else:
        header.set_intent(code)


def _set_dtype(
    header: nb.Nifti1Header,
    changed: tx.Dict[str, tx.Any],
    report: ConversionReport,
) -> None:
    """Write a data type; a type that NIfTI cannot store has no slot."""
    value = changed["dtype"]
    if value is None:
        report.approximated["dtype"] = "a NIfTI header always has a type"
        changed.pop("dtype")
        return
    try:
        header.set_data_dtype(value)
    except Exception:
        return
    changed.pop("dtype")
